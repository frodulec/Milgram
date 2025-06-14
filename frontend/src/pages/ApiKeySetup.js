import React from 'react';
import { Box, Heading } from '@chakra-ui/react';

function ApiKeySetup() {
  return (
    <Box p={6}>
      <Heading size="lg" mb={4}>
        API Key Setup
      </Heading>
      <p>Configure your API keys here.</p>
    </Box>
  );
}

export default ApiKeySetup;