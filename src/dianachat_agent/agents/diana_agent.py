from __future__ import annotations

import asyncio
import time
import logging
from typing import Optional, Dict, AsyncIterable, List, Tuple
import uuid

from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents import llm, stt, tokenize, transcription, vad, tts
from livekit.agents.pipeline import VoicePipelineAgent, AgentTranscriptionOptions
from livekit.plugins import openai, deepgram, silero, turn_detector, rag

from dianachat_agent.config.agent_config import AgentSettings
from dianachat_agent.rag.service import RAGService
from ..log import logger
import json
from dataclasses import replace

class DianaAgent:
    """Multimodal agent for handling voice and data interactions."""
    
    # Class-level constant
    LIVEKIT_CHAT_TOPIC = 'lk-chat-topic'
    
    def __init__(
            self,
            ctx: JobContext,
            settings: AgentSettings,
            vad=None,
            turn_detector_model=None,
        ):
        """Initialize the DianaChat agent.
        
        Args:
            ctx: LiveKit job context
            settings: Agent settings including API keys and configurations
            vad: Prewarmed VAD model
            turn_detector_model: Prewarmed turn detector model
        """
        self.ctx = ctx
        self.settings = settings
        self.response_cache = {}
        self.cache = None  # Initialize cache as None
        self._done = asyncio.Event()
        self._agent_started = False  # Track if agent has been started
        self._is_speaking = False  # Track if agent is currently speaking
        
        # Initialize RAG service
        self.rag_service = RAGService(settings=settings)
        self._rag_init_task = asyncio.create_task(self.rag_service.initialize())
        
        def replace_words(agent: VoicePipelineAgent, text: str | AsyncIterable[str]):
            """Replace words for better TTS pronunciation."""
            return tokenize.utils.replace_words(
                text=text, 
                replacements={
                    "sumsion": "Sumshun",
                    "Sumsion": "Sumshun",
                    "TX": "Texas",
                    "CISO": "C. I. S. O.", 
                    "CTO": "C.T.O.",
                    "CEO": "C.E.O.",
                    "pricelogic.ai": "price logic dot A.I.",  # Use dot A.I. instead of dot eye
                    "invest@pricelogic.ai": "invest at price logic dot A.I."
                }
            )
        
        # Use prewarmed VAD or create new one
        self.vad = vad if vad is not None else silero.VAD.load(
            min_speech_duration=0.25,    # Increased to reduce false positives
            min_silence_duration=1.0,    # Aligned with turn_hold_time
        )
        
        # Use prewarmed turn detector or create new one
        self.turn_detector = turn_detector_model if turn_detector_model is not None else turn_detector.EOUModel(
            unlikely_threshold=0.15,  # Default threshold for end-of-utterance detection
        )
        
        # Initialize STT with Deepgram
        self.stt = deepgram.STT(
            model=settings.deepgram_model,
            language=settings.deepgram_language,
            smart_format=True,
            punctuate=True,
            filler_words=True,
            keywords=[
                ("Sumsion", 3.0),     # Boost correct spelling even more
                ("Mike", 1.5),        # Boost recognition of first name
                ("Becky", 1.5),       # Boost recognition of first name
                ("PriceLogic", 2.0),  # Company name
            ],
        )
        
        # Initialize LLM with OpenAI implementation
        self.llm = openai.LLM(
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            parallel_tool_calls=True,
            api_key=settings.openai_api_key,
        )
        
        # Initialize TTS with OpenAI
        self.tts = openai.TTS(
            voice=settings.openai_voice,
            model="tts-1",
            api_key=settings.openai_api_key,
        )
        
        # Initialize chat context with system prompt
        chat_ctx = llm.ChatContext().append(text=settings.system_prompt, role="system")
        
        # Initialize the voice pipeline agent
        self.agent = VoicePipelineAgent(
            chat_ctx=chat_ctx,
            vad=self.vad,
            turn_detector=self.turn_detector,
            stt=self.stt,
            llm=self.llm,
            tts=self.tts,
            before_tts_cb=replace_words,
            before_llm_cb=self._enrich_with_rag,  # Add RAG enrichment callback
        )
        
        # Set up event handlers
        self.agent.on('error', self._on_error)
        self.agent.on('close', self._on_close)
        
        logger.info("DianaAgent initialized with LiveKit LLM, Deepgram STT, and OpenAI TTS")
    
    def _setup_event_handlers(self):
        """Set up event handlers for the room."""
        try:
            # Set up data message handler with proper task handling
            async def handle_data_message(event):
                try:
                    await self._on_data_received(event)
                except Exception as e:
                    logger.error(f"Error in data message handler: {e}", exc_info=True)
            
            def sync_handler(event):
                # Create task and add it to the event loop
                loop = asyncio.get_event_loop()
                task = loop.create_task(handle_data_message(event))
                # Optional: Store task reference if needed
                if not hasattr(self, '_pending_tasks'):
                    self._pending_tasks = set()
                self._pending_tasks.add(task)
                task.add_done_callback(self._pending_tasks.discard)
            
            self.ctx.room.on('data_received', sync_handler)
            logger.info("Event handlers set up successfully")
            
        except Exception as e:
            logger.error(f"Error setting up event handlers: {e}", exc_info=True)
            raise

    async def _cleanup_tasks(self):
        """Clean up any pending tasks."""
        if hasattr(self, '_pending_tasks'):
            pending_tasks = list(self._pending_tasks)
            if pending_tasks:
                logger.debug(f"Cleaning up {len(pending_tasks)} pending tasks")
                for task in pending_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            self._pending_tasks.clear()

    async def stop(self):
        """Stop the agent and clean up resources."""
        try:
            await self._cleanup_tasks()
            logger.info("Stopping agent...")
            self._done.set()
            
            if self.ctx and self.ctx.room:
                logger.info("Closing room connection...")
                await self.ctx.room.disconnect()
            
            if self.agent:
                logger.info("Stopping agent...")
                await self.agent.stop()
            
            logger.info("Agent stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping agent: {e}", exc_info=True)
            raise

    async def publish_data(self, data: Dict):
        """Publish data to the room using reliable data channel.
        
        The data should be a dictionary that may include:
        - text: The text content to send
        - timestamp: Message timestamp in milliseconds
        - type: Message type (e.g. "transcription")
        - final: Boolean indicating if this is a final transcription
        """
        try:
            # Add required fields for LiveKit chat format
            message_data = {
                "id": str(uuid.uuid4()),  # Generate unique ID
                "message": {  # Nest the full message data
                    "text": data.get("text", ""),
                    "type": data.get("type", "text"),
                    "final": data.get("final", True)
                },
                "timestamp": data.get("timestamp", int(time.time() * 1000)),
                "ignore": False  # Set to false so client displays it
            }
            
            logger.debug(f"Publishing data: {message_data}")
            # Use reliable delivery by default
            await self.ctx.room.local_participant.publish_data(
                json.dumps(message_data).encode('utf-8'),
                topic=DianaAgent.LIVEKIT_CHAT_TOPIC  # Use class constant
            )
            logger.debug("Data published successfully")
        except Exception as e:
            logger.error(f"Error publishing data: {e}", exc_info=True)
            raise

    async def _on_data_received(self, event: rtc.DataPacket) -> None:
        """Handle data messages from the room's data channel.
        
        The data packet may contain a JSON message with the following structure:
        {
            "id": str,
            "message": str | {"text": str, "speak": bool},
            "timestamp": int,
            "ignore": bool
        }
        
        If message is a JSON string, it will be parsed to extract text and speak flag.
        If not JSON or missing fields, defaults will be used.
        
        Args:
            event: LiveKit data packet containing the message
        """
        try:
            # Log raw event details
            logger.debug(f"Raw event data: {event.data!r}")
            logger.debug(f"Event topic: {event.topic}")
            logger.debug(f"Participant identity: {event.participant.identity}")
            
            # Parse outer JSON data
            data_json = json.loads(event.data.decode('utf-8'))
            logger.debug(f"Parsed outer JSON data: {data_json}")
            
            # Skip if message is from self
            if event.participant.identity == self.ctx.room.local_participant.identity:
                logger.debug(f"Skipping message from self: {event.data}")
                return

            # Override ignore flag for lk-chat-topic
            if event.topic == 'lk-chat-topic':
                data_json['ignore'] = False
                logger.debug("Overrode ignore flag for lk-chat-topic")
                logger.debug(f"Updated JSON data after override: {data_json}")
                
            # Skip if ignore flag is set
            if data_json.get('ignore', False):
                logger.debug(f"Skipping ignored message: {event.data}")
                return

            # Extract message and timestamp
            message = data_json.get('message')
            timestamp = data_json.get('timestamp')
            
            # Try to parse the message as JSON if it's a string
            speak = True  # Default speak value
            if isinstance(message, str):
                try:
                    inner_json = json.loads(message)
                    message = inner_json.get('text', message)
                    speak = inner_json.get('speak', True)
                    logger.debug(f"Parsed inner JSON - text: '{message}', speak: {speak}")
                except json.JSONDecodeError:
                    # Not a JSON string, use as is
                    logger.debug("Message is not JSON, using as plain text")
            
            # Log final parameters
            logger.debug("Final extracted parameters:")
            logger.debug(f"  message: '{message}'")
            logger.debug(f"  timestamp: {timestamp}")
            logger.debug(f"  speak: {speak}")
            logger.debug(f"  ignore: {data_json.get('ignore', False)}")

            if not message:
                logger.warning(f"Received data without message field: {event.data}")
                return

            # Process the message
            await self._process_message(
                message=message, 
                sender=event.participant.identity, 
                speak=speak,
                timestamp=timestamp
            )

        except json.JSONDecodeError:
            logger.error(f"Failed to parse data as JSON: {event.data}")
        except Exception as e:
            logger.error(f"Error processing data: {e}", exc_info=True)

    async def _process_message(
        self, 
        message: str, 
        sender: str, 
        speak: bool = True, 
        timestamp: int | None = None
    ) -> None:
        """Process a user message and generate an agent response.
        
        Args:
            message: The user's message text
            sender: Identity of the message sender
            speak: Whether to generate speech for the response (default: True)
            timestamp: Optional timestamp of the message
            
        Raises:
            Exception: If there is an error processing the message or generating response
        """
        try:
            logger.info(f"Processing message from {sender}")
            logger.debug(f"Message content: '{message}'")
            logger.debug(f"Speak enabled: {speak}")
            logger.debug(f"Timestamp: {timestamp}")
            
            # Only attempt interruption if agent is speaking
            if self._is_speaking:
                try:
                    # Check if agent exists, is started, and has interrupt method
                    if (hasattr(self, 'agent') and self.agent is not None and 
                        hasattr(self.agent, 'interrupt') and self._agent_started):
                        await self.agent.interrupt()
                        logger.info("Interrupted ongoing speech")
                    else:
                        logger.debug("Agent not ready for interruption")
                except AttributeError:
                    logger.warning("interrupt() method not available - requires recent version of livekit-agents")
                except Exception as e:
                    logger.error(f"Error interrupting speech: {e}")
            
            # Get the existing chat context from the voice pipeline agent
            chat_ctx = self.agent.chat_ctx
            logger.debug(f"Using voice pipeline chat context, length: {len(chat_ctx.messages)}")
            
            # Add the user's question for RAG search
            chat_ctx.messages.append(llm.ChatMessage(role="user", content=message))
            logger.debug(f"Added user message to context")
            
            # Get RAG enrichment
            enriched_message = await self._enrich_with_rag(self.agent, chat_ctx)
            logger.debug(f"Got RAG enrichment, length: {len(enriched_message) if enriched_message else 0}")
            
            # Build system message with RAG context
            if enriched_message:
                system_message = (
                    f"{self.settings.system_prompt}\n\n"
                    "IMPORTANT INSTRUCTION: You MUST use this verified information from "
                    "PriceLogic's official documentation to answer the user's question:\n\n"
                    f"{enriched_message}"
                )
                logger.debug(f"RAG context preview: {enriched_message[:200]}...")
                
                # Update the system message in the context
                for i, msg in enumerate(chat_ctx.messages):
                    if msg.role == "system":
                        chat_ctx.messages[i] = llm.ChatMessage(role="system", content=system_message)
                        logger.debug(f"Updated system message with length: {len(system_message)}")
                        break
            
            # Log the final context for debugging
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Final chat context being sent to LLM:")
                for i, msg in enumerate(chat_ctx.messages):
                    logger.debug(
                        f"Message {i + 1}:\n"
                        f"- Role: {msg.role}\n"
                        f"- Content preview: {msg.content[:200]}..."
                    )

            # Get response stream from LLM
            logger.debug("Getting response stream from LLM...")
            try:
                stream = self.agent.llm.chat(chat_ctx=chat_ctx)
                logger.debug("Successfully got LLM response stream")
            except openai.APIError as e:
                logger.error(f"OpenAI API error: {str(e)}")
                error_msg = "I apologize, but I'm having trouble connecting to my language model. Please try again in a moment."
                await self.publish_data({
                    "text": error_msg,
                    "type": "error",
                    "timestamp": int(time.time() * 1000)
                })
                return
            except openai.APITimeoutError as e:
                logger.error(f"OpenAI API timeout: {str(e)}")
                error_msg = "I apologize, but the request timed out. Please try again with a shorter message or in a moment."
                await self.publish_data({
                    "text": error_msg,
                    "type": "error",
                    "timestamp": int(time.time() * 1000)
                })
                return
            except Exception as e:
                logger.error(f"Error getting LLM response stream: {str(e)}", exc_info=True)
                error_msg = "I apologize, but I encountered an unexpected error. Please try again."
                await self.publish_data({
                    "text": error_msg,
                    "type": "error",
                    "timestamp": int(time.time() * 1000)
                })
                return
            
            # Process with or without speech
            try:
                if speak:
                    logger.info("Processing with speech enabled...")
                    self._is_speaking = True
                    response = await self.agent.say(stream, allow_interruptions=True)
                    logger.debug(f"Speech response completed: {response}")
                else:
                    logger.info("Processing in text-only mode...")
                    try:
                        class TextBuffer:
                            """Buffer for processing text chunks into complete words."""
                            def __init__(self):
                                self.current_word = []
                                self.response_text = []
                            
                            def process_chunk(self, chunk_text):
                                """Process a text chunk and return complete word if available."""
                                self.current_word.append(chunk_text)
                                
                                # If chunk ends with space or punctuation, we have a complete word
                                if chunk_text.endswith((' ', '.', '!', '?', '\n')):
                                    complete_word = ''.join(self.current_word)
                                    self.response_text.append(complete_word)
                                    self.current_word = []
                                    return complete_word
                                return None

                        response_text = ""
                        buffer = TextBuffer()  # Initialize buffer once before the loop
                        
                        async for chunk in stream:
                            # Log the chunk for debugging
                            logger.debug(f"Received chunk: {chunk}")
                            
                            # Extract content from chunk
                            chunk_text = None
                            if chunk.choices and len(chunk.choices) > 0:
                                delta = chunk.choices[0].delta
                                if delta and delta.content:
                                    chunk_text = delta.content
                            
                            if chunk_text:
                                response_text += chunk_text
                                # Use buffer to process chunks into complete words
                                complete_word = buffer.process_chunk(chunk_text)
                                
                                if complete_word:
                                    # Only publish complete words
                                    await self.publish_data({
                                        "text": complete_word,
                                        "timestamp": int(time.time() * 1000),
                                        "type": "transcription",
                                        "final": False
                                    })
                                    logger.debug(f"Published complete word: {complete_word}")
                        
                        # Handle any remaining partial word in buffer
                        if buffer.current_word:
                            final_word = ''.join(buffer.current_word)
                            await self.publish_data({
                                "text": final_word,
                                "timestamp": int(time.time() * 1000),
                                "type": "transcription",
                                "final": False
                            })
                            logger.debug(f"Published final partial word: {final_word}")
                        
                        # Publish final complete response
                        if response_text:
                            await self.publish_data({
                                "text": response_text,
                                "timestamp": int(time.time() * 1000),
                                "type": "transcription",
                                "final": True
                            })
                            logger.debug(f"Published complete response: {response_text}")
                        logger.debug("Text-only response completed")

                    except Exception as e:
                        logger.error(f"Error publishing text response: {e}", exc_info=True)
                        raise
            except Exception as e:
                logger.error(f"Error in response processing: {e}", exc_info=True)
                raise
            finally:
                self._is_speaking = False
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            raise

    async def _enrich_with_rag(self, agent: VoicePipelineAgent, context: agents.ChatContext) -> str:
        """Enrich user message with relevant context.
        
        Args:
            agent: The VoicePipelineAgent instance (unused but required for callback)
            context: The chat context containing the conversation history
            
        Returns:
            str: The enriched message with relevant context
        """
        try:
            # Get the last user message from context
            if not context or not context.messages:
                logger.warning("Empty chat context, cannot enrich")
                return ""
                
            # Find the last user message
            user_messages = [msg for msg in context.messages if msg.role == "user"]
            if not user_messages:
                logger.warning("No user messages in context")
                return ""
            
            # Get the most recent user message
            last_message = user_messages[-1]
            query = last_message.content if hasattr(last_message, 'content') else str(last_message)
            logger.debug(f"Using query for RAG: {query}")
            
            # Get relevant context from RAG
            try:
                enriched = await self.rag_service.enrich_context(query)
                if enriched:
                    logger.debug(f"Got RAG context: {enriched[:200]}...")
                    # Update the system message with the enriched context
                    if context.messages and context.messages[0].role == "system":
                        context.messages[0].content = f"{self.settings.system_prompt}\n\nRelevant Context:\n{enriched}"
                    return ""  # Return empty since we modified the system message directly
                else:
                    logger.warning("No relevant context found in RAG")
                    return ""
            except Exception as e:
                logger.error(f"Error getting RAG context: {str(e)}", exc_info=True)
                return ""
                
        except Exception as e:
            logger.error(f"Error in RAG enrichment: {str(e)}", exc_info=True)
            return ""

    def _get_cached_response(self, message: str) -> Optional[Dict]:
        """Get cached response if available and not expired."""
        if not self.settings.enable_response_caching:
            return None
            
        cache_key = message.strip().lower()
        cached = self.response_cache.get(cache_key)
        if not cached:
            return None
            
        timestamp, response = cached
        if time.time() - timestamp > self.settings.cache_ttl_seconds:
            del self.response_cache[cache_key]
            return None
            
        return response
    
    def _cache_response(self, message: str, response: Dict):
        """Cache a response with timestamp."""
        if not self.settings.enable_response_caching:
            return
            
        cache_key = message.strip().lower()
        self.response_cache[cache_key] = (time.time(), response)
    
    async def wait(self):
        """Wait for the agent to finish."""
        try:
            await self._done.wait()
        except asyncio.CancelledError:
            logger.info("Wait cancelled, initiating cleanup")
            await self.stop()
            raise

    def _on_track_published(self, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        """Handle new track publications.
        
        Track kinds:
        - 0: UNKNOWN
        - 1: AUDIO
        - 2: VIDEO
        - 3: DATA
        """
        try:
            logger.info(f"Track published: kind={publication.kind}, name={publication.name} from {participant.identity}")
            if publication.kind == 3:  # DATA track
                logger.info(f"Setting subscription for new DATA track from {participant.identity}")
                publication.set_subscribed(True)
                logger.info(f"Data track subscription status: {publication.subscribed}")
        except Exception as e:
            logger.error(f"Error handling track publication: {e}", exc_info=True)
    
    def _on_participant(self, participant: rtc.RemoteParticipant):
        """Handle new participant connections."""
        try:
            logger.info(f"New participant connected: {participant.identity}")
            logger.info(f"Participant tracks at connection: {[f'{t.name}(kind={t.kind})' for t in participant.track_publications.values()]}")
            
            # Subscribe to all tracks from this participant
            for track in participant.track_publications.values():
                # Track kinds: 0=UNKNOWN, 1=AUDIO, 2=VIDEO, 3=DATA
                logger.info(f"Checking new participant track: kind={track.kind}, name={track.name}")
                if track.kind == 3:  # DATA track
                    logger.info(f"Found DATA track from new participant {participant.identity} - attempting subscription")
                    track.set_subscribed(True)
                    logger.info(f"Data track subscription status: {track.subscribed}")
        except Exception as e:
            logger.error(f"Error handling participant connection: {e}", exc_info=True)
    
    def _on_track(self, track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        """Handle new tracks from participants.
        
        Track kinds:
        - 0: UNKNOWN
        - 1: AUDIO
        - 2: VIDEO
        - 3: DATA
        """
        try:
            logger.info(f"Track subscription event: kind={track.kind}, name={track.name} from {participant.identity}")
            logger.info(f"Publication status: subscribed={publication.subscribed}, name={publication.name}")
            
            if track.kind == 3:  # DATA track
                logger.info(f"Setting subscription for DATA track from {participant.identity}")
                publication.set_subscribed(True)
                logger.info(f"Data track subscription confirmed: {publication.subscribed}")
        except Exception as e:
            logger.error(f"Error handling track subscription: {e}", exc_info=True)

    def _on_error(self, error: Exception):
        """Handle agent errors."""
        logger.error(f"Agent error: {error}")
        if "stream closed" in str(error).lower():
            logger.info("Connection closed by server, initiating graceful shutdown")
        self._done.set()
    
    def _on_close(self):
        """Handle agent close."""
        logger.info("Agent closed")
        self._done.set()
    
    async def start(self):
        """Start the agent."""
        if self._agent_started:
            return
            
        # Wait for RAG initialization
        try:
            await self._rag_init_task
            logger.info("RAG service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            raise
        
        try:
            # Initialize RAG service first
            logger.info("Initializing RAG service...")
            # await self.rag_service.initialize()
            logger.info("RAG service initialized")
            
            # Set up event handlers
            self._setup_event_handlers()
            self.ctx.room.on('track_subscribed', self._on_track)
            self.ctx.room.on('track_published', self._on_track_published)
            self.ctx.room.on('participant_connected', self._on_participant)
            
            # Log current participants and their tracks
            logger.info(f"Room ID: {self.ctx.room.name}")
            logger.info(f"Local Participant: {self.ctx.room.local_participant.identity}")
            logger.info("Checking existing participants...")
            
            # Process any existing remote participants
            for participant in self.ctx.room.remote_participants.values():
                logger.info(f"Found existing participant: {participant.identity}")
                self._on_participant(participant)
                
            logger.info("Room setup complete")
            
            # Start the agent with the room context
            self.agent.start(room=self.ctx.room)
            self._agent_started = True  # Mark agent as started
            
            # Set up speech state handlers
            self.agent.on('agent_started_speaking', lambda: setattr(self, '_is_speaking', True))
            self.agent.on('agent_stopped_speaking', lambda: setattr(self, '_is_speaking', False))
            
            logger.info("Agent started successfully")

            # Send initial greeting
            try:
                await self.publish_data({
                    "text": self.settings.default_greeting,
                    "timestamp": int(time.time() * 1000),
                    "type": "text",
                    "final": True
                })
                # greeting in audio 
                # await self.agent.say(self.settings.default_greeting, allow_interruptions=True)
                logger.info("Initial greeting sent")
            except Exception as e:
                logger.error(f"Error sending initial greeting: {e}", exc_info=True)
            
            # Keep the agent running
            while not self._done.is_set():
                try:
                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    logger.info("Agent task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in agent loop: {e}", exc_info=True)
            
        except Exception as e:
            logger.error(f"Error during agent start: {e}", exc_info=True)
            raise
        finally:
            if self.ctx.room.connection_state == "connected":
                await self.stop()
                
class NoOpTTS:
    """TTS implementation that only publishes text without converting to speech."""
    
    def __init__(self, room: rtc.Room):
        self.room = room
        
    async def say(self, text: str) -> None:
        """Publish text without speech synthesis."""
        try:
            # Just publish the text directly
            data = {
                "text": text,
                "type": "text"
            }
            await self.room.local_participant.publish_data(
                json.dumps(data).encode(), 
                topic=DianaAgent.LIVEKIT_CHAT_TOPIC
            )
        except Exception as e:
            logger.error(f"Error in NoOpTTS say: {e}", exc_info=True)
            raise
