import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Box, VStack, Button, HStack, IconButton} from '@chakra-ui/react';
import { LuMoon, LuSun } from "react-icons/lu"
import { useColorMode } from "./components/ui/color-mode";
import { imageGenerator } from './services/imageGenerator';
import { fetchTTSAudio, getGameSequenceEventSource } from './services/apiService';
import ImageDisplay from './components/ImageDisplay';
import AudioControls from './components/AudioControls';
import MessageHistory from './components/MessageHistory';
import { useAudioPlayer } from './hooks/useAudioPlayer';

function App() {
  const { colorMode, toggleColorMode } = useColorMode();
  const [messages, setMessages] = useState([]);
  const [isStarted, setIsStarted] = useState(false);
  const [followCurrentMessage, setFollowCurrentMessage] = useState(true);

  const [syncQueue, setSyncQueue] = useState([]);
  const [currentSyncIndex, setCurrentSyncIndex] = useState(-1);
  const [isProcessingSync, setIsProcessingSync] = useState(false);

  const eventSourceRef = useRef(null);
  const messageRefs = useRef(new Map());

  // Use our audio player hook
  const {
    isPlaying,
    isMuted,
    volume,
    playbackRate,
    currentImage,
    audioRef,
    playNextItem,
    playPreviousItem,
    togglePlayPause,
    toggleMute,
    handleVolumeChange,
    handlePlaybackRateChange,
    setIsManuallyPaused,
    setCurrentImage
  } = useAudioPlayer({
    syncQueue,
    currentSyncIndex,
    setCurrentSyncIndex,
    isStarted
  });

  useEffect(() => {
    const loadDefaultImage = async () => {
      try {
        const imageBlob = await imageGenerator.generateImage({});
        const imageUrl = URL.createObjectURL(imageBlob);
        setCurrentImage(imageUrl);
      } catch (error) {
        console.error('Failed to generate default image:', error);
      }
    };
    
    loadDefaultImage();
    
    return () => {
      if (currentImage) {
        URL.revokeObjectURL(currentImage);
      }
    };
  }, []);

  const startExperience = () => {
    // If already started, just toggle play/pause
    if (isStarted) {
      togglePlayPause();
      return;
    }
    
    // Otherwise, start the experience
    setIsStarted(true);
    setIsManuallyPaused(false);
    eventSourceRef.current = getGameSequenceEventSource();

    eventSourceRef.current.onmessage = async (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'message') {
        setMessages(prev => [...prev, {
          speaker: data.speaker,
          text: data.text,
          timestamp: new Date()
        }]);
        addToSyncQueue(data.speaker, data.text);
      } else if (data.type === 'end') {
        eventSourceRef.current.close();
      }
    };

    eventSourceRef.current.onerror = (error) => {
      console.error('EventSource error:', error);
    };
  };

  const addToSyncQueue = (speaker, text) => {
    const syncItem = {
      id: Math.random().toString(36).substring(2) + Date.now().toString(36),
      speaker,
      text,
      audioUrl: null,
      imageUrl: null,
      isProcessed: false
    };
    setSyncQueue(prev => {
      const queue = Array.isArray(prev) ? prev : [];
      return [...queue, syncItem];
    });
  };

  const processSyncQueue = useCallback(async () => {
    if (isProcessingSync) return;
    setSyncQueue(prevQueue => {
      if (!Array.isArray(prevQueue)) {
        console.warn('syncQueue is not an array, initializing as empty array');
        return [];
      }
      const unprocessedItems = prevQueue.filter(item => !item.isProcessed);
      
      if (unprocessedItems.length === 0 || isProcessingSync) {
        return prevQueue;
      }
      
      const currentItem = unprocessedItems[0];
      setIsProcessingSync(true);
      
      Promise.all([
        processAudioForItem(currentItem),
        processImageForItem(currentItem)
      ]).then(() => {
        setSyncQueue(queue => 
          queue.map(item => 
            item.id === currentItem.id 
              ? { ...item, isProcessed: true }
              : item
          )
        );
        setIsProcessingSync(false);
      }).catch(error => {
        console.error('Failed to process sync item:', error);
        setIsProcessingSync(false);
      });
      
      return prevQueue;
    });
  }, [isProcessingSync]); 

  const processAudioForItem = async (item) => {
    if (item.text === 'ELECTRIC_SHOCK_IMAGE') {
      return;
    }
    
    try {
      const audioBlob = await fetchTTSAudio(item.speaker, item.text);
      const audioUrl = URL.createObjectURL(audioBlob);

      setSyncQueue(prev =>
        prev.map(queueItem =>
          queueItem.id === item.id
            ? { ...queueItem, audioUrl }
            : queueItem
        )
      );
    } catch (error) {
      console.error('Failed to generate TTS:', error);
    }
  };

  const processImageForItem = async (item) => {
    try {
      const params = {};
      if (item.speaker === 'Learner') {
        params.learner_message = item.text;
      } else if (item.speaker === 'Professor') {
        params.professor_message = item.text;
      } else if (item.speaker === 'Participant') {
        params.participant_message = item.text;
      } else if (item.speaker === 'SHOCKING_DEVICE') {
        params.display_shock = true;
      }

      const imageBlob = await imageGenerator.generateImage(params);
      const imageUrl = URL.createObjectURL(imageBlob);

      setSyncQueue(prev =>
        prev.map(queueItem =>
          queueItem.id === item.id
            ? { ...queueItem, imageUrl }
            : queueItem
        )
      );
    } catch (error) {
      console.error('Failed to generate game image:', error);
    }
  };

  useEffect(() => {
    if (followCurrentMessage && messageRefs.current.has(currentSyncIndex)) {
      const node = messageRefs.current.get(currentSyncIndex);
      if (node) {
        node.scrollIntoView({
          behavior: 'smooth',
          block: 'nearest',
        });
      }
    }
  }, [currentSyncIndex, followCurrentMessage, messages]);
  
  useEffect(() => {
    processSyncQueue();
  }, [syncQueue, processSyncQueue]);

  return (
    <Box p={4} maxW="1200px" mx="auto">
      <HStack justifyContent="space-between" alignItems="center" py={4} mb={4}>
        <Box textAlign="center">
          <Button
            colorScheme="brand"
            bg="brand.500"
            color="white"
            size="lg"
            onClick={startExperience}
          >
            {isStarted ? (isPlaying ? "Pause Experience" : "Resume Experience") : "Start Experience"}
          </Button>
        </Box>
        <IconButton
          onClick={toggleColorMode}
          variant="outline"
          size="sm"
        >
          {colorMode === "light" ? <LuSun /> : <LuMoon />}
        </IconButton>
      </HStack>
    
      <HStack align="flex-start" spacing={6}>
        {/* Left Column: Image and Audio Controls */}
        <VStack flex="1" align="stretch" spacing={4}>
          {/* Game Image Display with Placeholder */}
          <ImageDisplay currentImage={currentImage} colorMode={colorMode} />

          {/* Audio Controls (now under the image) */}
          <AudioControls
            isPlaying={isPlaying}
            isMuted={isMuted}
            volume={volume}
            playbackRate={playbackRate}
            currentSyncIndex={currentSyncIndex}
            totalItems={syncQueue.length}
            onPlayPause={togglePlayPause}
            onPrevious={playPreviousItem}
            onNext={playNextItem}
            onMute={toggleMute}
            onVolumeChange={handleVolumeChange}
            onPlaybackRateChange={handlePlaybackRateChange}
            colorMode={colorMode}
            isQueueEmpty={syncQueue.length === 0}
            hidePlayPauseButton={true}
          />
        </VStack>

        {/* Right Column: Message History */}
        <MessageHistory
          messages={messages}
          currentSyncIndex={currentSyncIndex}
          colorMode={colorMode}
          followCurrentMessage={followCurrentMessage}
          onToggleFollow={() => setFollowCurrentMessage(prev => !prev)}
        />
      </HStack>

      <audio ref={audioRef} />
    </Box>
  );
}

export default App;