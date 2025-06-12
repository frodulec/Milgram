import React from 'react';
import { Box, HStack, Text } from '@chakra-ui/react';

const MessageHistory = ({
  messages,
  currentSyncIndex,
  colorMode,
}) => {
  if (!Array.isArray(messages)) {
    // Log an error to help diagnose why messages might not be an array.
    // You could also return a placeholder UI here, like <Text>Loading messages...</Text>.
    console.error("MessageHistory: 'messages' prop is not an array. Received:", messages);
    return null; // Or return a fallback UI
  }

  return (
    <Box
      flex="1"
      bg={colorMode === 'light' ? "brand.50" : "gray.800"}
      p={4}
      borderRadius="md"
      maxH="600px"
      overflowY="auto"
      borderWidth="1px"
      borderColor="brand.500"
      boxShadow="md"
    >
      <HStack
        justifyContent="space-between"
        alignItems="center"
        mb={3}
        position="sticky"
        top="0"
        bg={colorMode === 'light' ? "brand.50" : "gray.800"}
        zIndex="1"
        p={2}
        borderTopRadius="md"
      >
        <Text
          fontSize="lg"
          fontWeight="bold"
          color={colorMode === 'light' ? "semantic.text" : "white"}
        >
          Conversation History
        </Text>
        </HStack>
      {messages.map((message, index) => {
        const isCurrentlyPlaying = index === currentSyncIndex;

        return (
          <Box
            key={index}
            mb={2}
            p={2}
            bg={isCurrentlyPlaying
              ? (colorMode === 'light' ? "brand.200" : "brand.900")
              : (colorMode === 'light' ? "white" : "gray.700")
            }
            borderRadius="md"
            borderLeft={isCurrentlyPlaying ? "4px solid" : "none"}
            borderLeftColor="brand.500"
            borderWidth="1px"
            borderColor={isCurrentlyPlaying ? "brand.500" : (colorMode === 'light' ? "gray.200" : "gray.600")}
            boxShadow={isCurrentlyPlaying ? "md" : "xs"}
            transition="all 0.2s ease-in-out"
          >
            <Text fontWeight="bold" color={
              message.speaker === 'Professor' ? (colorMode === 'dark' ? 'brand.300' : 'brand.500') :
                message.speaker === 'Learner' ? (colorMode === 'dark' ? 'teal.300' : 'teal.500') :
                  message.speaker === 'SHOCKING_DEVICE' ? (colorMode === 'dark' ? 'teal.300' : 'teal.500') :
                    (colorMode === 'dark' ? 'ruby.300' : 'ruby.500')
            }>
              {message.speaker === 'SHOCKING_DEVICE' ? 'Learner' : message.speaker}
            </Text>
            <Text color={colorMode === 'light' ? "semantic.text" : "white"}>
              {message.text === 'ELECTRIC_SHOCK_IMAGE' ?
                <Text as="span" color="voltage.danger">⚡ Electric Shock ⚡</Text> :
                message.text
              }
            </Text>
          </Box>
        );
      })}
    </Box>
  );
};

export default MessageHistory;