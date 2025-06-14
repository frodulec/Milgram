import React, { useRef, useEffect } from 'react';
import { Box, HStack, VStack, Text, IconButton } from '@chakra-ui/react';
import { LuPin, LuPinOff } from 'react-icons/lu';

const MessageHistory = ({
  messages,
  currentSyncIndex,
  colorMode,
  followCurrentMessage,
  onToggleFollow,
}) => {
  const messageRefs = useRef({});
  
  // Scroll to the current message when currentSyncIndex changes
  useEffect(() => {
    if (followCurrentMessage && 
        currentSyncIndex !== null && 
        currentSyncIndex >= 0 && 
        messageRefs.current[currentSyncIndex]) {
      messageRefs.current[currentSyncIndex].scrollIntoView({ 
        behavior: 'smooth', 
        block: 'center' // Center the message in the viewport
      });
    }
  }, [currentSyncIndex, followCurrentMessage]);

  if (!Array.isArray(messages)) {
    console.error("MessageHistory: 'messages' prop is not an array. Received:", messages);
    return null;
  }

  return (
    <VStack
      flex="1"
      bg={colorMode === 'light' ? "brand.50" : "gray.800"}
      p={4}
      borderRadius="md"
      height="600px"
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
        {/* Header content remains the same */}
        <Text
          fontSize="lg"
          fontWeight="bold"
          color={colorMode === 'light' ? "semantic.text" : "white"}
        >
          Conversation History
        </Text>
        <IconButton
          aria-label="Toggle auto-scroll"
          title={followCurrentMessage ? "Disable auto-scroll" : "Enable auto-scroll"}
          onClick={onToggleFollow}
          variant="outline"
          size="sm"
        >
          {followCurrentMessage ? <LuPinOff /> : <LuPin />}
        </IconButton>
      </HStack>
      <Box
      height={"100%"}
      width={"100%"}
      overflow={"auto"}>


      {messages.map((message, index) => {
        const isCurrentlyPlaying = index === currentSyncIndex;

        return (
          <Box
            key={index}
            ref={el => messageRefs.current[index] = el}
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
            {/* Message content remains the same */}
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
    </VStack>
  );
};

export default MessageHistory;