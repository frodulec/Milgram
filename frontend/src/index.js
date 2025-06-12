import React from 'react';
import ReactDOM from 'react-dom/client';
import { ChakraProvider } from '@chakra-ui/react';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import { system } from './theme'; // Make sure we're importing the system
import { ColorModeProvider } from './components/ui/color-mode';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <ColorModeProvider
      defaultTheme="light"
      enableSystem={false}
    >
      <ChakraProvider value={system}> {/* Make sure we're using value={system} */}
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </ChakraProvider>
    </ColorModeProvider>
  </React.StrictMode>
);