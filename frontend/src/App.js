import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Box, VStack, Text, Button, HStack, IconButton} from '@chakra-ui/react';
import { LuMoon, LuSun } from "react-icons/lu"
import { useColorMode } from "./components/ui/color-mode";
import { imageGenerator } from './services/imageGenerator';
import { fetchTTSAudio, getGameSequenceEventSource } from './services/apiService';
import ImageDisplay from './components/ImageDisplay';
import AudioControls from './components/AudioControls';
import MessageHistory from './components/MessageHistory';

function App() {
  const { colorMode, toggleColorMode } = useColorMode();
  const [messages, setMessages] = useState([]);
  const [currentImage, setCurrentImage] = useState(null);
  const [isStarted, setIsStarted] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [volume, setVolume] = useState(1);
  const [playbackRate, setPlaybackRate] = useState(2);
  const [followCurrentMessage, setFollowCurrentMessage] = useState(true);

  const [syncQueue, setSyncQueue] = useState([]);
  const [currentSyncIndex, setCurrentSyncIndex] = useState(0);
  const [isProcessingSync, setIsProcessingSync] = useState(false);
  
  const audioRef = useRef(null);
  const eventSourceRef = useRef(null);
  const messageRefs = useRef(new Map());
  const startExperience = () => {
    setIsStarted(true);
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

  const playNextItem = useCallback(() => {
    const nextIndex = currentSyncIndex + 1;

    if (nextIndex < syncQueue.length) {
      const nextItem = syncQueue[nextIndex];
      if (nextItem && nextItem.isProcessed) {
        setCurrentSyncIndex(nextIndex);
        setIsPlaying(false);
      }
    } else {
      console.log('No more items in sync queue');
      setIsPlaying(false);
    }
  }, [syncQueue, currentSyncIndex]);

  const playCurrentItem = useCallback(() => {
    if (syncQueue.length === 0 || currentSyncIndex < 0 || currentSyncIndex >= syncQueue.length) {
      return;
    }

    const currentItem = syncQueue[currentSyncIndex];

    if (!currentItem || !currentItem.isProcessed || isPlaying) {
      return;
    }

    setIsPlaying(true);

    if (currentItem.imageUrl) {
      setCurrentImage(prevImage => {
        if (prevImage && prevImage !== currentItem.imageUrl) {
          URL.revokeObjectURL(prevImage);
        }
        return currentItem.imageUrl;
      });
    }

    if (currentItem.audioUrl && audioRef.current) {
      const audio = audioRef.current;
      audio.pause();
      audio.currentTime = 0;
      audio.src = currentItem.audioUrl;
      audio.play().catch(error => {
        console.error('Failed to play audio:', error);
        setIsPlaying(false);
        playNextItem();
      });
    } else {
      if (currentItem.speaker === 'SHOCKING_DEVICE') {
        setTimeout(() => {
          setIsPlaying(false);
        playNextItem();
        }, 1300);
      } else {
          setIsPlaying(false);
        playNextItem();
      }
    }
}, [syncQueue, currentSyncIndex, isPlaying, playNextItem]);

  const playPreviousItem = useCallback(() => {
    if (currentSyncIndex > 0) {
      setCurrentSyncIndex(currentSyncIndex - 1);
      setIsPlaying(false);
    }
  }, [currentSyncIndex]);

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
    if (!isPlaying && syncQueue.length > 0 && currentSyncIndex >= 0 && currentSyncIndex < syncQueue.length) {
      const currentItem = syncQueue[currentSyncIndex];
      if (currentItem && currentItem.isProcessed && isStarted) {
        playCurrentItem();
      }
    }
}, [syncQueue, currentSyncIndex, isPlaying, playCurrentItem, isStarted]);

  useEffect(() => {
    processSyncQueue();
  }, [syncQueue, processSyncQueue]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleLoadedData = () => {
      audio.volume = isMuted ? 0 : volume;
      audio.playbackRate = playbackRate;
    };

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => {
      if (!audio.ended) {
        setIsPlaying(false);
      }
    };

    const handleEnded = () => {
      setIsPlaying(false);
      playNextItem();
    };

    audio.addEventListener('loadeddata', handleLoadedData);
    audio.addEventListener('play', handlePlay);
    audio.addEventListener('pause', handlePause);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('loadeddata', handleLoadedData);
      audio.removeEventListener('play', handlePlay);
      audio.removeEventListener('pause', handlePause);
      audio.removeEventListener('ended', handleEnded);
    };
  }, [volume, isMuted, playbackRate, playNextItem]);

  const togglePlayPause = () => {
    if (!isStarted) {
      setIsStarted(true);
    }

    const audio = audioRef.current;
    if (!audio || currentSyncIndex === -1) return;

    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
    } else {
      audio.play().catch(error => {
        console.error('Failed to play audio:', error);
        setIsPlaying(false);
      });
    }
  };

  const toggleMute = () => {
    setIsMuted(!isMuted);
    if (audioRef.current) {
      audioRef.current.volume = isMuted ? volume : 0;
    }
  };

  const handleVolumeChange = (newVolume) => {
    setVolume(newVolume);
    if (audioRef.current && !isMuted) {
      audioRef.current.volume = newVolume;
    }
  };

  const handlePlaybackRateChange = (newRate) => {
    setPlaybackRate(newRate);
    if (audioRef.current) {
      audioRef.current.playbackRate = newRate;
    }
  };

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
            {isStarted ? "Restart Experience" : "Start Experience"}
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
