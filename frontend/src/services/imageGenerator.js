import {
  StaticCanvas,
  Rect,
  Polygon,
  Group,
  Circle,
  FabricText,
  FabricImage
} from 'fabric';
import professorSprite from '../assets/professor_w.png';
import learnerSprite from '../assets/learner.png';
import participantSprite from '../assets/student.png';
import shockSprite from '../assets/electricity.png';
import backgroundImage from '../assets/background.jpg'
import speechBubble from '../assets/speech_bubble.png';

export class ImageGenerator {
  constructor() {
    const canvasEl = document.createElement('canvas');
    canvasEl.width = 1024;
    canvasEl.height = 1024;
    
    this.canvas = new StaticCanvas(canvasEl, {
      width: 1024,
      height: 1024,
      backgroundColor: '#F0F0F0'
    });
    
    this.professorPos = { x: 700, y: 780 };
    this.professorScale = { x: 0.7, y: 0.7 };

    this.shockPos = { x: 700, y: 400 };
    this.shockScale = { x: 0.075, y: 0.075 };

    this.learnerPos = { x: 710, y: 390 };
    this.learnerScale = { x: 1.05, y: 1.05 };

    this.participantPos = { x: 400, y: 780 };
    this.participantScale = { x: 0.17, y: 0.17 };
    
    this.sprites = {};
    this.loadSprites();
  }

  async loadSprites() {
    const loadImage = async (url, x_scale = 0.5, y_scale = 0.5) => {
      return new Promise((resolve, reject) => {
        // Add a timeout to prevent hanging indefinitely
        const timeoutId = setTimeout(() => {
          console.warn(`Image load timed out for: ${url}`);
          reject(new Error(`Timeout loading image at ${url}`));
        }, 10000); // 10 second timeout

        try {
          const img = new Image();

          img.onload = () => {
          clearTimeout(timeoutId);

            const fabricImg = new FabricImage(img, {
              centeredRotation: true,
              centeredScaling: true,
              scaleX: x_scale,  
              scaleY: y_scale,
              perPixelTargetFind: false
      });

            // Check if the image has valid dimensions
            if (fabricImg.width === 0 || fabricImg.height === 0) {
              console.warn(`Image has invalid dimensions: ${url}`, fabricImg.width, fabricImg.height);
              reject(new Error(`Image has invalid dimensions at ${url}`));
              return;
            }

            resolve(fabricImg);
    };

          img.onerror = (error) => {
            clearTimeout(timeoutId);
            console.error(`Failed to load image at ${url}`, error);
            reject(new Error(`Failed to load image at ${url}`));
          };

          img.crossOrigin = 'anonymous';
          img.src = url;
    } catch (error) {
          clearTimeout(timeoutId);
          console.error(`Exception during image load for: ${url}`, error);
          reject(error);
    }
      });
    };

    try {
      this.sprites.professor = await loadImage(professorSprite, this.professorScale.x, this.professorScale.y);
      this.sprites.learner = await loadImage(learnerSprite, this.learnerScale.x, this.learnerScale.y);
      this.sprites.participant = await loadImage(participantSprite, this.participantScale.x, this.participantScale.y);
      this.sprites.shock = await loadImage(shockSprite, this.shockScale.x, this.shockScale.y);
      this.sprites.background = await loadImage(backgroundImage, 1, 1);
      this.sprites.speechBubble = await loadImage(speechBubble, 0.15, 0.15);

    } catch (error) {
      console.error('Failed to load sprites:', error);
  }
  }

  
async generateImage(params) {
  // Wait for sprites to load if they haven't already
  try {
    if (!this.sprites.professor || !this.sprites.learner || !this.sprites.participant) {
      await this.loadSprites();
    }

    this.canvas.clear();
    
    if (this.sprites.background) {
      this.sprites.background.set({
        left: 0,
        top: 0,
        originX: 'left',
        originY: 'top',
        width: this.canvas.width,
        height: this.canvas.height
      });
      this.canvas.add(this.sprites.background);
    } else {
      // Fallback to color if background image failed to load
      this.canvas.backgroundColor = '#F0F0F0';
    }

    // Draw characters
    await this.drawCharacters();

    // Add speech bubbles based on messages
    if (params.professor_message) {
      await this.addSpeechBubble(params.professor_message, 'right');
    }
    if (params.participant_message) {
      await this.addSpeechBubble(params.participant_message, 'left');
    }
    if (params.learner_message) {
      await this.addSpeechBubble(params.learner_message, 'up');
    }

    // Handle shock display
    if (params.display_shock) {
      await this.addShockEffect();
    }

    return new Promise((resolve) => {
      this.canvas.renderAll();
      this.canvas.lowerCanvasEl.toBlob(resolve);
    });
  } catch (error) {
    console.error('Error generating image:', error);
    
    // Return a fallback solid color image when generation fails
    const fallbackCanvas = document.createElement('canvas');
    fallbackCanvas.width = 1024;
    fallbackCanvas.height = 1024;
    const ctx = fallbackCanvas.getContext('2d');
    ctx.fillStyle = '#F0F0F0';
    ctx.fillRect(0, 0, 1024, 1024);
    
    return new Promise((resolve) => {
      fallbackCanvas.toBlob(resolve);
    });
  }
}

  async drawCharacters() {
  try {
    // Professor
    if (this.sprites.professor) {
      const professor = this.sprites.professor;
      professor.set({
        left: this.professorPos.x - professor.getScaledWidth() / 2,
        top: this.professorPos.y - professor.getScaledHeight() / 2
      });
      this.canvas.add(professor);
    }
    
    // Learner
    if (this.sprites.learner) {
      const learner = this.sprites.learner;
      learner.set({
        left: this.learnerPos.x - learner.getScaledWidth() / 2,
        top: this.learnerPos.y - learner.getScaledHeight() / 2
      });
      this.canvas.add(learner);
    }
    
    // Participant
    if (this.sprites.participant) {
      const participant = this.sprites.participant;
      participant.set({
        left: this.participantPos.x - participant.getScaledWidth() / 2,
        top: this.participantPos.y - participant.getScaledHeight() / 2
      });
      this.canvas.add(participant);
    }
  } catch (error) {
    console.error('Error drawing characters:', error);
    // Continue execution rather than throwing, so we can still return a partial image
  }
}

async addSpeechBubble(text, direction) {
  try {
    if (!this.sprites.speechBubble) {
      console.error('Speech bubble image not loaded');
      return;
    }

    // Text configuration
    const fontSize = 16;
    const fontFamily = 'Liberation Mono';
    const padding = 20; // Padding inside the bubble
    
    // Additional offsets to account for the bubble's shape
    const topOffset = 0;  // Increase top padding
    const bottomOffset = 20;  // Space to leave at the bottom
    
    // Create a temporary text object to calculate character metrics
    const tempTextObj = new FabricText(text, {
      fontSize: fontSize,
      fontFamily: fontFamily,
      fill: '#000000',
    });
    
    // Calculate text metrics
    const avgCharWidth = tempTextObj.width / text.length;
    const lineHeight = fontSize * 1.2; // 120% of font size for line spacing
    
    // Target aspect ratio (width:height)
    const desiredAspectRatio = 3.0 / 2.0;

    // Calculate optimal width for text wrap
    const totalChars = text.length;
    const optimalCharsPerLine = Math.sqrt(
        totalChars * desiredAspectRatio * lineHeight / avgCharWidth
    );
    
    // Calculate target line width and ensure minimum width
    const targetLineWidth = Math.max(20, Math.floor(optimalCharsPerLine));

    // Wrap text using a custom wrapper function
    const wrappedLines = this.wrapText(text, targetLineWidth);

    // Calculate actual dimensions of wrapped text
    const actualTextWidth = wrappedLines.reduce((max, line) =>
      Math.max(max, line.length * avgCharWidth), 0);
    const actualTextHeight = wrappedLines.length * lineHeight;
    
    // Get base dimensions of speech bubble
    const baseWidth = this.sprites.speechBubble.getScaledWidth();
    const baseHeight = this.sprites.speechBubble.getScaledHeight();

    // Calculate bubble dimensions with padding
    const targetWidth = Math.max(actualTextWidth + padding * 2, baseWidth);
    const targetHeight = Math.max(actualTextHeight + padding * 2 + topOffset + bottomOffset, baseHeight);
    
    const bubble = await this.resizeSpeechBubble(targetWidth, targetHeight);

    let left, top, originX, originY, flip;
    switch (direction) {
      case 'right':
        // Anchor at bottom right
        left = 650;
        top = 650;
        originX = 'right';
        originY = 'bottom';
        flip = true;
        break;
      case 'left':
        // Anchor at bottom left
        left = 440;
        top = 650;
        originX = 'left';
        originY = 'bottom';
        flip = false;
        break;
      case 'up':
        left = 520;
        top = 160;
        originX = 'left'; // Assuming top-left for 'up' for now
        originY = 'top';
        flip = true;
        break;
    }

    // Flip the bubble if needed
    if (flip) {
      bubble.set('flipX', true);
    }

    // Position bubble with appropriate origin
    bubble.set({
      left,
      top,
      originX,
      originY
    });

    // Add bubble to canvas
    this.canvas.add(bubble);

    // Calculate the absolute top-left coordinates of the bubble
    // These are the coordinates where the bubble's visual top-left corner is on the canvas
    let bubbleAbsoluteLeft = left;
    if (originX === 'right') {
      bubbleAbsoluteLeft = left - targetWidth;
    } else if (originX === 'center') { // For completeness, though not used for bubble origin yet
      bubbleAbsoluteLeft = left - targetWidth / 2;
    }
    
    let bubbleAbsoluteTop = top;
    if (originY === 'bottom') {
      bubbleAbsoluteTop = top - targetHeight;
    } else if (originY === 'center') { // For completeness
      bubbleAbsoluteTop = top - targetHeight / 2;
    }

    // Calculate the position for the text block
    // Text objects have originX: 'center', originY: 'top'
    // The horizontal center for the text lines:
    const textBlockHorizontalCenter = bubbleAbsoluteLeft + targetWidth / 2;
    // The vertical start for the first line of text (top edge of the text block):
    const textBlockVerticalStart = bubbleAbsoluteTop + padding + topOffset;
    // Add text lines with adjusted positioning
    wrappedLines.forEach((line, index) => {
      const textObj = new FabricText(line, {
        fontSize: fontSize,
        fontFamily: fontFamily,
        fill: '#000000',
        left: textBlockHorizontalCenter, // Centered horizontally within the bubble
        top: textBlockVerticalStart + (index * lineHeight), // Positioned from the top of the text area
        originX: 'center', // Text line itself is centered on its 'left'
        originY: 'top'    // Text line 'top' refers to its actual top
      });
      this.canvas.add(textObj);
    });

  } catch (error) {
    console.error('Error adding speech bubble:', error);
  }
}

async resizeSpeechBubble(targetWidth, targetHeight) {
  const original = this.sprites.speechBubble;
  const baseWidth = original.getScaledWidth();
  const baseHeight = original.getScaledHeight();

  // If no resize needed, return a clone of the original
  if (targetWidth <= baseWidth && targetHeight <= baseHeight) {
    return original.clone();
  }

  // Calculate middle points for slicing
  const midX = Math.floor(baseWidth / 2);
  const midY = Math.floor(baseHeight / 2);

  // Create a new canvas with target dimensions
  const canvas = document.createElement('canvas');
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const ctx = canvas.getContext('2d');

  // Create a temporary canvas to draw the scaled original image
  const tempOriginalCanvas = document.createElement('canvas');
  tempOriginalCanvas.width = baseWidth;
  tempOriginalCanvas.height = baseHeight;
  const tempOriginalCtx = tempOriginalCanvas.getContext('2d');
  
  // Draw the original element with its current scale to our temp canvas
  const originalElement = original.getElement();
  tempOriginalCtx.drawImage(
    originalElement, 
    0, 0, originalElement.width, originalElement.height,
    0, 0, baseWidth, baseHeight
  );
  
  // Calculate new corner positions
  const rightX = targetWidth - (baseWidth - midX);
  const bottomY = targetHeight - (baseHeight - midY);
  
  // Draw the corners using our scaled source image
  // Top-left corner
  ctx.drawImage(tempOriginalCanvas, 0, 0, midX, midY, 0, 0, midX, midY);
  
  // Top-right corner
  ctx.drawImage(
    tempOriginalCanvas, 
    midX, 0, baseWidth - midX, midY,
    rightX, 0, baseWidth - midX, midY
  );
  
  // Bottom-left corner
  ctx.drawImage(
    tempOriginalCanvas, 
    0, midY, midX, baseHeight - midY,
    0, bottomY, midX, baseHeight - midY
  );
  
  // Bottom-right corner
  ctx.drawImage(
    tempOriginalCanvas, 
    midX, midY, baseWidth - midX, baseHeight - midY,
    rightX, bottomY, baseWidth - midX, baseHeight - midY
  );
  
  // Get a sample from the center for the expanded middle area
  const centerSampleCanvas = document.createElement('canvas');
  centerSampleCanvas.width = 1;
  centerSampleCanvas.height = 1;
  const centerSampleCtx = centerSampleCanvas.getContext('2d');
  centerSampleCtx.drawImage(tempOriginalCanvas, midX, midY, 1, 1, 0, 0, 1, 1);
  const centerPixelData = centerSampleCtx.getImageData(0, 0, 1, 1).data;
  
  // Draw stretched middle parts from our scaled source
  // Top middle section
  ctx.drawImage(
    tempOriginalCanvas,
    midX, 0, 1, midY,
    midX, 0, rightX - midX, midY
  );
  
  // Bottom middle section
  ctx.drawImage(
    tempOriginalCanvas,
    midX, midY, 1, baseHeight - midY,
    midX, bottomY, rightX - midX, baseHeight - midY
  );
  
  // Left middle section
  ctx.drawImage(
    tempOriginalCanvas,
    0, midY, midX, 1,
    0, midY, midX, bottomY - midY
  );
  
  // Right middle section
  ctx.drawImage(
    tempOriginalCanvas,
    midX, midY, baseWidth - midX, 1,
    rightX, midY, baseWidth - midX, bottomY - midY
  );
  
  // Fill the center expanded area
  ctx.fillStyle = `rgba(${centerPixelData[0]}, ${centerPixelData[1]}, ${centerPixelData[2]}, ${centerPixelData[3]/255})`;
  ctx.fillRect(midX, midY, rightX - midX, bottomY - midY);
  
  // Create a new Fabric image from the canvas
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const fabricImg = new FabricImage(img, {
        centeredRotation: true,
        centeredScaling: true,
        scaleX: 1,
        scaleY: 1,
        width: targetWidth,
        height: targetHeight,
        perPixelTargetFind: false
      });
      
      resolve(fabricImg);
    };
    img.onerror = (err) => {
      console.error('Error loading resized image:', err);
      // Fall back to original if there's an error
      resolve(original.clone());
    };
    img.src = canvas.toDataURL();
  });
}

  // Add this new method to handle text wrapping
  wrapText(text, maxCharsPerLine) {
    const words = text.split(' ');
    const lines = [];
    let currentLine = '';

    words.forEach(word => {
      // Check if adding this word would exceed the max line length
      const testLine = currentLine ? `${currentLine} ${word}` : word;

      if (testLine.length <= maxCharsPerLine) {
        currentLine = testLine;
      } else {
        // If the current line is already at the limit, add it and start a new line
        lines.push(currentLine);
        currentLine = word;

        // Handle case where a single word is longer than maxCharsPerLine
        while (currentLine.length > maxCharsPerLine) {
          lines.push(currentLine.substring(0, maxCharsPerLine));
          currentLine = currentLine.substring(maxCharsPerLine);
        }
      }
    });

    // Don't forget the last line
    if (currentLine) {
      lines.push(currentLine);
    }

    // Handle the edge case of empty input
    if (lines.length === 0) {
      lines.push('');
    }

    return lines;
  }


  async addShockEffect() {
    if (!this.sprites.shock) return;

    try {
      const shock = this.sprites.shock;
    shock.set({
        left: this.learnerPos.x - shock.getScaledWidth() / 2,
        top: this.learnerPos.y - shock.getScaledHeight() / 2,
      opacity: 0.8
    });

    // Add glow effect
    const glow = new Circle({
      left: this.learnerPos.x - 50,
      top: this.learnerPos.y - 50,
      radius: 50,
      fill: 'rgba(255, 255, 0, 0.2)',
      stroke: 'rgba(255, 255, 0, 0.5)',
      strokeWidth: 3
    });

    this.canvas.add(shock);
    } catch (error) {
      console.error('Error adding shock effect:', error);
  }
}
}

export const imageGenerator = new ImageGenerator();