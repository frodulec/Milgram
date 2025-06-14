import { useState, useRef, useCallback, useEffect } from 'react';
import { getGameSequenceEventSource } from '../services/apiService';

export function useExperienceManager({ resetQueue, resetAudioPlayer, addToSyncQueue }) {
  const [messages, setMessages] = useState([]);
  const [isStarted, setIsStarted] = useState(false);
  const [followCurrentMessage, setFollowCurrentMessage] = useState(true);

  const eventSourceRef = useRef(null);
  const messageRefs = useRef(new Map());

  const startExperience = useCallback(() => {
    // Reset all relevant states for a clean start
    setMessages([]);
    resetQueue();
    resetAudioPlayer();
    setFollowCurrentMessage(true);

    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    setIsStarted(true);
    eventSourceRef.current = getGameSequenceEventSource();

    eventSourceRef.current.onmessage = async (event) => {
      console.log('Received event data:', event.data);
      const data = JSON.parse(event.data);
      console.log('Parsed event data:', data);

      if (data.type === 'message') {
        console.log('Processing message from:', data.speaker);
        setMessages(prev => [...prev, {
          speaker: data.speaker,
          text: data.text,
          timestamp: new Date()
        }]);
        addToSyncQueue(data.speaker, data.text);
      } else if (data.type === 'end') {
        console.log('Server indicated end of experience with reason:', data.reason || 'No reason provided');
        eventSourceRef.current.close();
        setIsStarted(false);
      } else {
        console.log('Unknown event type:', data.type);
      }
    };

    eventSourceRef.current.onerror = (error) => {
      console.error('EventSource error:', error);
      eventSourceRef.current.close();
      setIsStarted(false);
    };
  }, [resetQueue, resetAudioPlayer, addToSyncQueue]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  return {
    messages,
    isStarted,
    followCurrentMessage,
    messageRefs,
    startExperience,
    setFollowCurrentMessage
  };
}