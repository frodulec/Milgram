import axios from 'axios';

const apiClient = axios.create({
  baseURL: 'http://127.0.0.1:8000' // Replace 8000 with your backend port
});

export const fetchTTSAudio = async (speaker, text) => {
  const response = await apiClient.post('/api/tts', {
    role: speaker,
    message: text
  }, {
    responseType: 'blob'
  });
  return response.data; // This is the blob
};

export const getGameSequenceEventSource = () => {
  return new EventSource('http://localhost:8000/api/game-sequence-example');
};

export const getNewContersationEventSource = () => {
  return new EventSource('http://localhost:8000/api/run-experiment');
};