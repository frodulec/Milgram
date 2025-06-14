import React, { useState } from 'react';
import {
  Box,
  VStack,
  HStack,
  Text,
  Switch,
  Select,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Button,
  Textarea,
  Input,
  Divider,
  useToast,
  Alert,
  AlertIcon,
  FormControl,
  FormLabel,
  FormHelperText,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
} from '@chakra-ui/react';
import { useSettings } from '../contexts/SettingsContext';
import { storageService } from '../services/storageService';

function Settings() {
  const { settings, updateSettings, clearApiKey } = useSettings();
  const [localSettings, setLocalSettings] = useState(settings);
  const [newTemplate, setNewTemplate] = useState('');
  const toast = useToast();

  const handleSave = () => {
    updateSettings(localSettings);
    toast({
      title: 'Settings saved',
      status: 'success',
      duration: 2000,
      isClosable: true,
    });
  };

  const handleReset = () => {
    const defaultSettings = {
      theme: 'system',
      fontSize: 'medium',
      autoRefreshRate: 5000,
      soundEnabled: true,
      voltageSettings: {
        startingVoltage: 0,
        incrementAmount: 15,
      },
      messageTemplates: [
        'Please continue with the next question.',
        'That is correct.',
        'That is incorrect, please try again.',
        'Let me explain this concept further.',
      ],
      characterProfiles: {
        student: {
          name: 'Student',
          persona: 'Eager to learn but sometimes makes mistakes',
        },
        professor: {
          name: 'Professor',
          persona: 'Knowledgeable and patient teacher',
        },
      },
      experimentParameters: {
        questionDifficulty: 'medium',
        responseTimeLimit: 30,
      },
    };
    setLocalSettings(defaultSettings);
  };

  const addTemplate = () => {
    if (newTemplate.trim()) {
      setLocalSettings({
        ...localSettings,
        messageTemplates: [...localSettings.messageTemplates, newTemplate.trim()],
      });
      setNewTemplate('');
    }
  };

  const removeTemplate = (index) => {
    setLocalSettings({
      ...localSettings,
      messageTemplates: localSettings.messageTemplates.filter((_, i) => i !== index),
    });
  };

  const exportData = () => {
    const sessionData = storageService.getSessionData();
    const exportData = {
      settings: localSettings,
      sessionData,
      exportDate: new Date().toISOString(),
    };
    storageService.exportData(exportData, 'experiment-settings.json');
    toast({
      title: 'Data exported',
      description: 'Settings and session data exported successfully',
      status: 'success',
      duration: 3000,
      isClosable: true,
    });
  };

  return (
    <Box maxW="800px" mx="auto">
      <VStack spacing={6} align="stretch">
        <Text fontSize="2xl" fontWeight="bold">
          Settings
        </Text>

        <Accordion allowMultiple defaultIndex={[0]}>
          {/* General Settings */}
          <AccordionItem>
            <AccordionButton>
              <Box flex="1" textAlign="left" fontWeight="bold">
                General Settings
              </Box>
              <AccordionIcon />
            </AccordionButton>
            <AccordionPanel pb={4}>
              <VStack spacing={4} align="stretch">
                <FormControl>
                  <FormLabel>Theme</FormLabel>
                  <Select
                    value={localSettings.theme}
                    onChange={(e) =>
                      setLocalSettings({ ...localSettings, theme: e.target.value })
                    }
                  >
                    <option value="light">Light</option>
                    <option value="dark">Dark</option>
                    <option value="system">System</option>
                  </Select>
                </FormControl>

                <FormControl>
                  <FormLabel>Font Size</FormLabel>
                  <Select
                    value={localSettings.fontSize}
                    onChange={(e) =>
                      setLocalSettings({ ...localSettings, fontSize: e.target.value })
                    }
                  >
                    <option value="small">Small</option>
                    <option value="medium">Medium</option>
                    <option value="large">Large</option>
                  </Select>
                </FormControl>

                <FormControl>
                  <FormLabel>Auto Refresh Rate (ms)</FormLabel>
                  <NumberInput
                    value={localSettings.autoRefreshRate}
                    onChange={(value) =>
                      setLocalSettings({
                        ...localSettings,
                        autoRefreshRate: parseInt(value),
                      })
                    }
                    min={1000}
                    max={60000}
                    step={1000}
                  >
                    <NumberInputField />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                  <FormHelperText>How often to refresh game state</FormHelperText>
                </FormControl>

                <FormControl display="flex" alignItems="center">
                  <FormLabel mb="0">Sound Enabled</FormLabel>
                  <Switch
                    isChecked={localSettings.soundEnabled}
                    onChange={(e) =>
                      setLocalSettings({ ...localSettings, soundEnabled: e.target.checked })
                    }
                  />
                </FormControl>
              </VStack>
            </AccordionPanel>
          </AccordionItem>

          {/* Voltage Settings */}
          <AccordionItem>
            <AccordionButton>
              <Box flex="1" textAlign="left" fontWeight="bold">
                Voltage Settings
              </Box>
              <AccordionIcon />
            </AccordionButton>
            <AccordionPanel pb={4}>
              <VStack spacing={4} align="stretch">
                <FormControl>
                  <FormLabel>Starting Voltage</FormLabel>
                  <NumberInput
                    value={localSettings.voltageSettings.startingVoltage}
                    onChange={(value) =>
                      setLocalSettings({
                        ...localSettings,
                        voltageSettings: {
                          ...localSettings.voltageSettings,
                          startingVoltage: parseInt(value),
                        },
                      })
                    }
                    min={0}
                    max={100}
                  >
                    <NumberInputField />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                </FormControl>

                <FormControl>
                  <FormLabel>Increment Amount</FormLabel>
                  <NumberInput
                    value={localSettings.voltageSettings.incrementAmount}
                    onChange={(value) =>
                      setLocalSettings({
                        ...localSettings,
                        voltageSettings: {
                          ...localSettings.voltageSettings,
                          incrementAmount: parseInt(value),
                        },
                      })
                    }
                    min={5}
                    max={50}
                    step={5}
                  >
                    <NumberInputField />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                  <FormHelperText>Voltage increase per button press</FormHelperText>
                </FormControl>
              </VStack>
            </AccordionPanel>
          </AccordionItem>

          {/* Message Templates */}
          <AccordionItem>
            <AccordionButton>
              <Box flex="1" textAlign="left" fontWeight="bold">
                Message Templates
              </Box>
              <AccordionIcon />
            </AccordionButton>
            <AccordionPanel pb={4}>
              <VStack spacing={4} align="stretch">
                <Text fontSize="sm" color="gray.600">
                  Pre-defined messages for quick access
                </Text>
                
                {localSettings.messageTemplates.map((template, index) => (
                  <HStack key={index}>
                    <Input value={template} isReadOnly flex="1" />
                    <Button
                      size="sm"
                      colorScheme="red"
                      onClick={() => removeTemplate(index)}
                    >
                      Remove
                    </Button>
                  </HStack>
                ))}

                <HStack>
                  <Input
                    placeholder="Add new template..."
                    value={newTemplate}
                    onChange={(e) => setNewTemplate(e.target.value)}
                    flex="1"
                  />
                  <Button size="sm" colorScheme="blue" onClick={addTemplate}>
                    Add
                  </Button>
                </HStack>
              </VStack>
            </AccordionPanel>
          </AccordionItem>

          {/* Character Profiles */}
          <AccordionItem>
            <AccordionButton>
              <Box flex="1" textAlign="left" fontWeight="bold">
                Character Profiles
              </Box>
              <AccordionIcon />
            </AccordionButton>
            <AccordionPanel pb={4}>
              <VStack spacing={4} align="stretch">
                <FormControl>
                  <FormLabel>Student Name</FormLabel>
                  <Input
                    value={localSettings.characterProfiles.student.name}
                    onChange={(e) =>
                      setLocalSettings({
                        ...localSettings,
                        characterProfiles: {
                          ...localSettings.characterProfiles,
                          student: {
                            ...localSettings.characterProfiles.student,
                            name: e.target.value,
                          },
                        },
                      })
                    }
                  />
                </FormControl>

                <FormControl>
                  <FormLabel>Student Persona</FormLabel>
                  <Textarea
                    value={localSettings.characterProfiles.student.persona}
                    onChange={(e) =>
                      setLocalSettings({
                        ...localSettings,
                        characterProfiles: {
                          ...localSettings.characterProfiles,
                          student: {
                            ...localSettings.characterProfiles.student,
                            persona: e.target.value,
                          },
                        },
                      })
                    }
                    rows={3}
                  />
                </FormControl>

                <FormControl>
                  <FormLabel>Professor Name</FormLabel>
                  <Input
                    value={localSettings.characterProfiles.professor.name}
                    onChange={(e) =>
                      setLocalSettings({
                        ...localSettings,
                        characterProfiles: {
                          ...localSettings.characterProfiles,
                          professor: {
                            ...localSettings.characterProfiles.professor,
                            name: e.target.value,
                          },
                        },
                      })
                    }
                  />
                </FormControl>

                <FormControl>
                  <FormLabel>Professor Persona</FormLabel>
                  <Textarea
                    value={localSettings.characterProfiles.professor.persona}
                    onChange={(e) =>
                      setLocalSettings({
                        ...localSettings,
                        characterProfiles: {
                          ...localSettings.characterProfiles,
                          professor: {
                            ...localSettings.characterProfiles.professor,
                            persona: e.target.value,
                          },
                        },
                      })
                    }
                    rows={3}
                  />
                </FormControl>
              </VStack>
            </AccordionPanel>
          </AccordionItem>
        </Accordion>

        <Divider />

        {/* Action Buttons */}
        <HStack spacing={4} justify="flex-end">
          <Button onClick={handleReset} variant="outline">
            Reset to Defaults
          </Button>
          <Button onClick={exportData} variant="outline">
            Export Data
          </Button>
          <Button onClick={handleSave} colorScheme="blue">
            Save Settings
          </Button>
        </HStack>

        <Alert status="info">
          <AlertIcon />
          Changes are saved locally and will persist between sessions.
        </Alert>
      </VStack>
    </Box>
  );
}

export default Settings;