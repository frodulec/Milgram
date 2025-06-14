import { useState, useCallback, useEffect } from 'react';
import { fetchTTSAudio } from '../services/apiService';
import { imageGenerator } from '../services/imageGenerator';

export function useSyncQueue() {
  const [syncQueue, setSyncQueue] = useState([]);
  const [currentSyncIndex, setCurrentSyncIndex] = useState(-1);
  const [isProcessingSync, setIsProcessingSync] = useState(false);

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

  // Effect to process the sync queue
  useEffect(() => {
    processSyncQueue();
  }, [syncQueue, processSyncQueue]);

  const resetQueue = () => {
    setSyncQueue([]);
    setCurrentSyncIndex(-1);
    setIsProcessingSync(false);
  };

  return {
    syncQueue,
    currentSyncIndex,
    setCurrentSyncIndex,
    addToSyncQueue,
    resetQueue
  };
}