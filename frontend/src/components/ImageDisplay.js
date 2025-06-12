import React from 'react';
import { Box, Image, Text } from '@chakra-ui/react';

const ImageDisplay = ({ currentImage, colorMode }) => {
  return (
    <Box
      textAlign="center"
      height="600px"
      width="100%"
      bg={colorMode === 'light' ? "brand.50" : "gray.800"}
      borderRadius="md"
      display="flex"
      alignItems="center"
      justifyContent="center"
      overflow="hidden"
      borderWidth="2px"
      borderColor="brand.500"
      boxShadow="md"
    >
      {currentImage ? (
        <Box
          width="100%"
          height="100%"
          display="flex"
          alignItems="center"
          justifyContent="center"
        >
          <Image
            src={currentImage}
            alt="Game Scene"
            width="auto"
            height="auto"
            maxH="100%"
            maxW="100%"
            objectFit="scale-down"
            margin="auto"
          />
        </Box>
      ) : (
        <Text color="semantic.textMuted">Waiting for image...</Text>
      )}
    </Box>
  );
};

export default ImageDisplay;