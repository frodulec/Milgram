import React from 'react';
import { Box, HStack, IconButton, Slider, Text, Button} from '@chakra-ui/react';
import { FaPlay, FaPause, FaVolumeUp, FaVolumeMute, FaForward, FaBackward } from 'react-icons/fa';

const AudioControls = ({
  isPlaying,
  isStarted,
  runPlaybackFunc,
  isMuted,
  volume,
  playbackRate,
  currentSyncIndex,
  totalItems,
  onPlayPause,
  onPrevious,
  onNext,
  onMute,
  onVolumeChange,
  onPlaybackRateChange,
  colorMode,
  isQueueEmpty,
  hidePlayPauseButton = false
}) => {

  return (
    <Box
      bg={colorMode === 'light' ? "brand.100" : "gray.700"}
      p={4}
      borderRadius="md"
      borderWidth="1px"
      borderColor="brand.500"
      boxShadow="sm"
    >
      <Text
        fontSize="md"
        fontWeight="bold"
        mb={2}
        textAlign="center"
        color={colorMode === 'light' ? "semantic.text" : "white"}
      >
       Playback Controls
      </Text>

      {/* Playback Controls */}
      <HStack spacing={2} mb={3}>
        <IconButton
          size="sm"
          onClick={onPrevious}
          isDisabled={currentSyncIndex <= 0}
          aria-label="Previous"
          colorScheme="brand"
        >
          <FaBackward />
        </IconButton>

            <Box textAlign="center">
              <Button
                colorScheme="brand"
                bg="brand.500"
                color="white"
                size="sm"
                onClick={runPlaybackFunc}
              >
                {isStarted ? (isPlaying ? "Pause Experiment" : "Resume Experiment") : "Start Experiment"}
              </Button>
            </Box>

        {!hidePlayPauseButton && (
          <IconButton
            size="sm"
            onClick={onPlayPause}
            isDisabled={currentSyncIndex === -1}
            aria-label={isPlaying ? "Pause" : "Play"}
            colorScheme="brand"
          >
            {isPlaying ? <FaPause /> : <FaPlay />}
          </IconButton>
        )}
        <IconButton
          size="sm"
          onClick={onNext}
          isDisabled={currentSyncIndex >= totalItems - 1}
          aria-label="Next"
          colorScheme="brand"
        >
          <FaForward />
        </IconButton>
        <IconButton
          size="sm"
          onClick={onMute}
          aria-label={isMuted ? "Unmute" : "Mute"}
          colorScheme="brand"
        >
          {isMuted ? <FaVolumeMute /> : <FaVolumeUp />}
        </IconButton>

      </HStack>

      {/* Volume Control */}
      <HStack spacing={2} mb={2}>
        <Text
          fontSize="sm"
          minW="60px"
          color={colorMode === 'light' ? "semantic.text" : "white"}
        >
          Volume:
        </Text>
        <Slider.Root
          value={[volume]}
          onValueChange={(details) => onVolumeChange(details.value[0])}
          min={0}
          max={1}
          step={0.1}
          flex={1}
        >
          <Slider.Control>
            <Slider.Track>
              <Slider.Range />
            </Slider.Track>
            <Slider.Thumb>
              <Slider.HiddenInput />
            </Slider.Thumb>
          </Slider.Control>
        </Slider.Root>
        <Text
          fontSize="sm"
          minW="30px"
          color={colorMode === 'light' ? "semantic.text" : "white"}
        >
          {Math.round(volume * 100)}%
        </Text>
      </HStack>

      {/* Playback Speed Control */}
      <HStack spacing={2}>
        <Text
          fontSize="sm"
          minW="60px"
          color={colorMode === 'light' ? "semantic.text" : "white"}
        >
          Speed:
        </Text>
        <Slider.Root
          value={[playbackRate]}
          onValueChange={(details) => onPlaybackRateChange(details.value[0])}
          min={0.5}
          max={4}
          step={0.1}
          flex={1}
        >
          <Slider.Control>
            <Slider.Track>
              <Slider.Range />
            </Slider.Track>
            <Slider.Thumb>
              <Slider.HiddenInput />
            </Slider.Thumb>
          </Slider.Control>
        </Slider.Root>
        <Text
          fontSize="sm"
          minW="30px"
          color={colorMode === 'light' ? "semantic.text" : "white"}
        >
          {playbackRate}x
        </Text>
      </HStack>
    </Box>
  );
};

export default AudioControls;