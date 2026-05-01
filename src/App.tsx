import React, { useState, useRef, useEffect } from 'react';
import { Upload, X, Image as ImageIcon, ThumbsUp, ThumbsDown, Loader2, Sparkles, User, Trash2, ArrowRight, Info, Key, Focus, Copy, Palette, Check, Sliders, Eye, ScanFace } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { GoogleGenAI } from '@google/genai';
import { FaceLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';

declare global {
  interface Window {
    aistudio?: {
      hasSelectedApiKey: () => Promise<boolean>;
      openSelectKey: () => Promise<void>;
    };
  }
}

// Local Storage Hook
function useLocalStorage<T>(key: string, initialValue: T) {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.log(error);
      return initialValue;
    }
  });

  const setValue = (value: T | ((val: T) => T)) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.log(error);
    }
  };

  return [storedValue, setValue] as const;
}

const cropFace = (img: HTMLImageElement, rect: {x: number, y: number, width: number, height: number}) => {
  const canvas = document.createElement('canvas');
  const padding = rect.width * 0.3;
  const x = Math.max(0, rect.x - padding);
  const y = Math.max(0, rect.y - padding);
  const width = Math.min(img.width - x, rect.width + padding * 2);
  const height = Math.min(img.height - y, rect.height + padding * 2);

  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx?.drawImage(img, x, y, width, height, 0, 0, width, height);
  return canvas.toDataURL('image/jpeg');
};

// File Dropzone Component
function FileDropzone({ onDrop, multiple = false, children, className = "" }: { onDrop: (files: File[]) => void, multiple?: boolean, children: React.ReactNode, className?: string }) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onDrop(multiple ? Array.from(e.dataTransfer.files) : [e.dataTransfer.files[0]]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onDrop(multiple ? Array.from(e.target.files) : [e.target.files[0]]);
    }
  };

  return (
    <div
      className={`relative border-2 border-dashed rounded-2xl transition-all cursor-pointer flex flex-col items-center justify-center overflow-hidden ${isDragging ? 'border-indigo-500 bg-indigo-500/10 scale-[1.02]' : 'border-zinc-800 hover:border-zinc-700 bg-zinc-900/30'} ${className}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current?.click()}
    >
      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        multiple={multiple}
        accept="image/*, image/heic, image/heif"
        onChange={handleChange}
      />
      {children}
    </div>
  );
}

export default function App() {
  const [hasKey, setHasKey] = useState<boolean | null>(null);
  const [activeTab, setActiveTab] = useState<'profile' | 'studio' | 'style'>('profile');
  
  // Persisted State
  const [selfImages, setSelfImages] = useLocalStorage<string[]>('faceswap_self_images', []);
  const [likedImages, setLikedImages] = useLocalStorage<string[]>('faceswap_liked_images', []);
  const [dislikedFeedback, setDislikedFeedback] = useLocalStorage<string[]>('faceswap_disliked_feedback', []);

  // Studio State
  const [targetImage, setTargetImage] = useState<{url: string, aspectRatio: string} | null>(null);
  const [targetImgDimensions, setTargetImgDimensions] = useState<{width: number, height: number} | null>(null);
  const [faceRect, setFaceRect] = useState<{x: number, y: number, width: number, height: number} | null>(null);
  const [croppedFaceUrl, setCroppedFaceUrl] = useState<string | null>(null);
  
  const [instructions, setInstructions] = useState('');
  const [faceAdjustments, setFaceAdjustments] = useState({ scale: 100, jawline: 0, nose: 0, eyes: 0 });
  const [showWireframe, setShowWireframe] = useState(true);
  const [bestMatchIndex, setBestMatchIndex] = useState<number | null>(null);

  const [isGenerating, setIsGenerating] = useState(false);
  const [isAnalyzingPose, setIsAnalyzingPose] = useState(false);
  const [results, setResults] = useState<string[]>([]);
  const [recreationPrompt, setRecreationPrompt] = useState<string>('');
  const [copiedRecreation, setCopiedRecreation] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Style Extractor State
  const [styleImage, setStyleImage] = useState<string | null>(null);
  const [stylePrompt, setStylePrompt] = useState<string>('');
  const [isAnalyzingStyle, setIsAnalyzingStyle] = useState(false);
  const [copiedStyle, setCopiedStyle] = useState(false);

  // MediaPipe State
  const [faceLandmarker, setFaceLandmarker] = useState<FaceLandmarker | null>(null);
  const targetImageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const checkKey = async () => {
      if (window.aistudio?.hasSelectedApiKey) {
        const has = await window.aistudio.hasSelectedApiKey();
        setHasKey(has);
      } else {
        setHasKey(true); // Fallback if not in AI Studio environment
      }
    };
    checkKey();

    const initMediaPipe = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        const landmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU"
          },
          outputFaceBlendshapes: true,
          runningMode: "IMAGE",
          numFaces: 1
        });
        setFaceLandmarker(landmarker);
      } catch (e) {
        console.error("Failed to initialize MediaPipe", e);
      }
    };
    initMediaPipe();
  }, []);

  const detectAndDrawLandmarks = () => {
    if (!faceLandmarker || !targetImageRef.current || !canvasRef.current || !showWireframe) return;
    const img = targetImageRef.current;
    const canvas = canvasRef.current;
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    try {
      const results = faceLandmarker.detect(img);
      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        const drawingUtils = new DrawingUtils(ctx);
        for (const landmarks of results.faceLandmarks) {
          drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#C0C0C070", lineWidth: 1 });
          drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#FF3030" });
          drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: "#FF3030" });
          drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#30FF30" });
          drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: "#30FF30" });
          drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#E0E0E0" });
          drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: "#E0E0E0" });
        }
      }
    } catch (e) {
      console.error("Landmark detection failed", e);
    }
  };

  useEffect(() => {
    if (targetImage && showWireframe) {
      // Small delay to ensure image is rendered before detecting
      setTimeout(detectAndDrawLandmarks, 100);
    } else if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  }, [targetImage, showWireframe, faceLandmarker]);

  // Handlers
  const handleSelfImagesDrop = (files: File[]) => {
    files.forEach(file => {
      const reader = new FileReader();
      reader.onloadend = () => {
        setSelfImages(prev => {
          const newImages = [...prev, reader.result as string];
          return newImages.slice(-10); // Allow up to 10 identity photos
        });
      };
      reader.readAsDataURL(file);
    });
  };

  const removeSelfImage = (index: number) => {
    setSelfImages(prev => prev.filter((_, i) => i !== index));
  };

  const handleTargetImageDrop = (files: File[]) => {
    if (files.length > 0) {
      setFaceRect(null);
      setCroppedFaceUrl(null);
      setBestMatchIndex(null);
      const reader = new FileReader();
      reader.onloadend = () => {
        const img = new Image();
        img.onload = async () => {
          let ratio = "1:1";
          const ar = img.width / img.height;
          if (ar > 1.5) ratio = "16:9";
          else if (ar > 1.1) ratio = "4:3";
          else if (ar < 0.6) ratio = "9:16";
          else if (ar < 0.9) ratio = "3:4";
          
          setTargetImgDimensions({ width: img.width, height: img.height });
          setTargetImage({ url: reader.result as string, aspectRatio: ratio });

          // Try basic face detection for bounding box
          if ('FaceDetector' in window) {
            try {
              const detector = new (window as any).FaceDetector();
              const faces = await detector.detect(img);
              if (faces.length > 0) {
                faces.sort((a: any, b: any) => (b.boundingBox.width * b.boundingBox.height) - (a.boundingBox.width * a.boundingBox.height));
                const rect = faces[0].boundingBox;
                setFaceRect(rect);
                setCroppedFaceUrl(cropFace(img, rect));
              }
            } catch (e) {
              console.warn("FaceDetector failed", e);
            }
          }
        };
        img.src = reader.result as string;
      };
      reader.readAsDataURL(files[0]);
    }
  };

  const handleImageClick = (e: React.MouseEvent<HTMLImageElement>) => {
    e.stopPropagation();
    e.preventDefault();
    if (!targetImgDimensions || !targetImage) return;
    
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const scaleX = targetImgDimensions.width / rect.width;
    const scaleY = targetImgDimensions.height / rect.height;
    const imgX = x * scaleX;
    const imgY = y * scaleY;

    const boxSize = Math.min(targetImgDimensions.width, targetImgDimensions.height) * 0.25;
    const newRect = {
      x: Math.max(0, imgX - boxSize/2),
      y: Math.max(0, imgY - boxSize/2),
      width: boxSize,
      height: boxSize
    };
    setFaceRect(newRect);

    const img = new Image();
    img.onload = () => {
      setCroppedFaceUrl(cropFace(img, newRect));
    };
    img.src = targetImage.url;
  };

  const handleAnalyzeStyle = async () => {
    if (!styleImage) return;
    setIsAnalyzingStyle(true);
    setStylePrompt('');
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || process.env.GEMINI_API_KEY });
      const response = await ai.models.generateContent({
        model: 'gemini-3.1-pro-preview',
        contents: {
          parts: [
            {
              inlineData: {
                data: styleImage.split(',')[1],
                mimeType: styleImage.split(';')[0].split(':')[1]
              }
            },
            {
              text: `Analyze this image and write a highly detailed image generation prompt that captures its exact style, lighting, camera angle, mood, and composition. This prompt will be used to generate a new image with a different person's face, so describe the subject generically (e.g., 'a person', 'a man', 'a woman') but be extremely specific about the environment, aesthetics, and photography style. Output ONLY the prompt text, ready to be copy-pasted into an AI image generator.`
            }
          ]
        }
      });
      setStylePrompt(response.text || '');
    } catch (err: any) {
      console.error(err);
      if (err.message?.includes('Requested entity was not found') || err.message?.includes('API key not valid')) {
        setHasKey(false);
      }
    } finally {
      setIsAnalyzingStyle(false);
    }
  };

  const handleGenerate = async () => {
    if (selfImages.length === 0) {
      setError("Please upload at least one reference photo of yourself in the Identity tab.");
      setActiveTab('profile');
      return;
    }
    if (!targetImage) {
      setError("Please upload a target photo.");
      return;
    }

    setIsGenerating(true);
    setIsAnalyzingPose(true);
    setError(null);
    setResults([]);
    setRecreationPrompt('');

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || process.env.GEMINI_API_KEY });

      // Step 1: Find best reference image based on pose
      let bestIdx = 0;
      if (selfImages.length > 1) {
        try {
          const parts: any[] = [
            { text: "Image 1 is the TARGET scene. The following images (2 to N) are REFERENCE photos of a person. Analyze the 3D head pose (yaw, pitch, roll) of the main subject in Image 1. Then, find the REFERENCE photo that has the most similar head pose. Return ONLY the number (2 to N) of the best matching reference photo. Do not return any other text." },
            { inlineData: { data: targetImage.url.split(',')[1], mimeType: targetImage.url.split(';')[0].split(':')[1] } }
          ];
          selfImages.forEach(img => {
            parts.push({ inlineData: { data: img.split(',')[1], mimeType: img.split(';')[0].split(':')[1] } });
          });
          const response = await ai.models.generateContent({ model: 'gemini-3.1-flash-preview', contents: { parts } });
          const match = response.text?.match(/\d+/);
          if (match) {
            bestIdx = parseInt(match[0]) - 2;
            bestIdx = Math.max(0, Math.min(bestIdx, selfImages.length - 1));
          }
        } catch (e) {
          console.warn("Failed to find best reference, defaulting to first", e);
        }
      }
      setBestMatchIndex(bestIdx);
      setIsAnalyzingPose(false);

      const generatePrompt = async () => {
        const parts = [
          {
            inlineData: {
              data: targetImage.url.split(',')[1],
              mimeType: targetImage.url.split(';')[0].split(':')[1]
            }
          },
          {
            text: `Analyze this target scene. Write a highly detailed image generation prompt that would recreate this exact scene, lighting, composition, and mood. Describe the main subject generically (e.g., "a person") but incorporate these specific user instructions regarding the subject's pose or expression: "${instructions || 'Keep the original pose and expression'}". The output should be JUST the prompt text, ready to be copy-pasted into an AI image generator.`
          }
        ];
        const response = await ai.models.generateContent({
          model: 'gemini-3.1-pro-preview',
          contents: { parts }
        });
        return response.text || '';
      };

      const generateSingle = async (index: number) => {
        const parts: any[] = [];
        
        // Use best image first, then up to 3 others
        const bestImage = selfImages[bestIdx];
        const otherImages = selfImages.filter((_, i) => i !== bestIdx).slice(0, 3);
        const imagesToUse = [bestImage, ...otherImages];

        imagesToUse.forEach(img => {
          parts.push({
            inlineData: {
              data: img.split(',')[1],
              mimeType: img.split(';')[0].split(':')[1]
            }
          });
        });

        parts.push({
          inlineData: {
            data: targetImage.url.split(',')[1],
            mimeType: targetImage.url.split(';')[0].split(':')[1]
          }
        });

        if (croppedFaceUrl) {
          parts.push({
            inlineData: {
              data: croppedFaceUrl.split(',')[1],
              mimeType: croppedFaceUrl.split(';')[0].split(':')[1]
            }
          });
        }

        const likedImagesToUse = likedImages.slice(-2);
        likedImagesToUse.forEach(img => {
          parts.push({
            inlineData: {
              data: img.split(',')[1],
              mimeType: img.split(';')[0].split(':')[1]
            }
          });
        });

        let prompt = `You are an expert AI photo editor and high-end VFX specialist.\n\nINPUTS:\n- Images 1 to ${imagesToUse.length}: Reference photos of the USER. Image 1 is the BEST MATCH for the target head pose.\n- Image ${imagesToUse.length + 1}: The TARGET SCENE.\n`;
        
        let nextIndex = imagesToUse.length + 2;
        if (croppedFaceUrl) {
          prompt += `- Image ${nextIndex}: A cropped highlight of the specific subject in the TARGET SCENE that needs to be recast.\n`;
          nextIndex++;
        }

        if (likedImagesToUse.length > 0) {
          prompt += `- Images ${nextIndex} to ${nextIndex + likedImagesToUse.length - 1}: Previous generations the user liked (use for style reference).\n`;
        }

        prompt += `\nYOUR TASK:\nRecast the main subject in the TARGET SCENE as the USER. Do NOT just paste a face. You must completely integrate the USER'S head and facial features onto the subject's body so it looks like the USER was the original model for the photo.`;
        if (croppedFaceUrl) {
          prompt += ` Specifically, target the subject shown in Image ${imagesToUse.length + 2}.`;
        }
        
        prompt += `\n\nCRITICAL CONSTRAINTS:\n`;
        prompt += `1. ANATOMY & PERSPECTIVE (CRUCIAL): The USER'S head MUST match the exact 3D angle (yaw, pitch, roll) of the original subject. Image 1 is the BEST REFERENCE for this pose. Scale the head and facial features to be perfectly proportional to the body. The neck, jawline, and hairline must connect seamlessly without looking like a "photoshopped head".\n`;
        prompt += `2. IDENTITY: The new face MUST strongly resemble the USER from the reference photos, adapting their features naturally to the new angle and expression.\n`;
        prompt += `3. LIGHTING, SHADOWS & SKIN TONE: Analyze the lighting direction, intensity, color temperature, and shadows on the original subject. Apply this exact lighting map to the USER'S face. Match the skin tone and texture of the USER'S face to the rest of the subject's exposed body (neck, hands, etc.).\n`;
        prompt += `4. EXPRESSION & EMOTION: Detect the facial expression of the original subject. Subtly adjust the USER'S features to convey this exact same mood.\n`;
        prompt += `5. CONTEXT: Maintain the original background, clothing, pose, and overall style of the TARGET SCENE perfectly. Do not alter the environment.\n`;

        // Apply Face Adjustments
        if (faceAdjustments.scale !== 100 || faceAdjustments.jawline !== 0 || faceAdjustments.nose !== 0 || faceAdjustments.eyes !== 0) {
          prompt += `\nUSER ADJUSTMENTS (MUST APPLY):\nThe user has requested the following structural modifications to their facial features in the final output:\n`;
          if (faceAdjustments.scale !== 100) prompt += `- Head Scale: ${faceAdjustments.scale}% (Scale the overall head size relative to the target body)\n`;
          if (faceAdjustments.jawline !== 0) prompt += `- Jawline Width: ${faceAdjustments.jawline > 0 ? 'Wider/Stronger' : 'Narrower/Softer'} by ${Math.abs(faceAdjustments.jawline)}%\n`;
          if (faceAdjustments.nose !== 0) prompt += `- Nose Size: ${faceAdjustments.nose > 0 ? 'Larger' : 'Smaller'} by ${Math.abs(faceAdjustments.nose)}%\n`;
          if (faceAdjustments.eyes !== 0) prompt += `- Eye Size: ${faceAdjustments.eyes > 0 ? 'Larger' : 'Smaller'} by ${Math.abs(faceAdjustments.eyes)}%\n`;
          prompt += `Apply these structural adjustments while maintaining photorealism and the user's core identity.\n`;
        }

        if (instructions) {
          prompt += `\nADVANCED USER INSTRUCTIONS (MUST FOLLOW):\n"${instructions}"\n`;
          prompt += `You MUST interpret and apply these instructions (e.g., changing expression to smiling, altering the pose to look left, or adjusting lighting) while still maintaining the user's core identity and the overall scene context. If the instructions contradict the original scene (e.g., "look left" when the original looks right), prioritize the USER INSTRUCTIONS and adjust the head angle and lighting accordingly.\n`;
        }
        if (dislikedFeedback.length > 0) {
          prompt += `\nTHINGS TO STRICTLY AVOID (Based on past feedback):\n${dislikedFeedback.slice(-3).join(', ')}\n`;
        }

        prompt += `\nGenerate variation ${index + 1} ensuring all constraints are met perfectly.`;

        parts.push({ text: prompt });

        const response = await ai.models.generateContent({
          model: 'gemini-3.1-flash-image-preview',
          contents: { parts },
          config: {
            imageConfig: {
              aspectRatio: targetImage.aspectRatio,
              imageSize: "1K"
            }
          }
        });

        for (const part of response.candidates?.[0]?.content?.parts || []) {
          if (part.inlineData) {
            return `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`;
          }
        }
        throw new Error("No image generated");
      };

      const promises = [generateSingle(0), generateSingle(1), generateSingle(2), generatePrompt()];
      const [res1, res2, res3, promptRes] = await Promise.all(promises);
      
      setResults([res1 as string, res2 as string, res3 as string]);
      setRecreationPrompt(promptRes as string);
    } catch (err: any) {
      console.error(err);
      if (err.message?.includes('Requested entity was not found') || err.message?.includes('API key not valid')) {
        setHasKey(false);
        setError("API Key session expired or invalid. Please select your API key again.");
      } else {
        setError(err.message || "Failed to generate images. Please try again.");
      }
    } finally {
      setIsGenerating(false);
      setIsAnalyzingPose(false);
    }
  };

  const handleThumbsUp = (img: string) => {
    if (!likedImages.includes(img)) {
      setLikedImages(prev => [...prev, img]);
    }
  };

  const handleThumbsDown = (img: string) => {
    const reason = window.prompt("What didn't you like about this image? (This helps improve future results)");
    if (reason) {
      setDislikedFeedback(prev => [...prev, reason]);
    }
  };

  if (hasKey === false) {
    return (
      <div className="min-h-screen bg-zinc-950 text-zinc-50 font-sans flex items-center justify-center p-6">
        <div className="glass-panel p-8 rounded-2xl max-w-md w-full text-center space-y-6">
          <div className="w-16 h-16 bg-indigo-500/20 text-indigo-400 rounded-full flex items-center justify-center mx-auto mb-4">
            <Key className="w-8 h-8" />
          </div>
          <h2 className="text-2xl font-semibold">API Key Required</h2>
          <p className="text-zinc-400">
            To use the advanced Gemini 3.1 Flash Image model for high-quality face swapping, lighting matching, and expression detection, you need to select a paid Google Cloud API key.
          </p>
          <p className="text-sm text-zinc-500">
            <a href="https://ai.google.dev/gemini-api/docs/billing" target="_blank" rel="noreferrer" className="text-indigo-400 hover:underline">Learn more about billing</a>
          </p>
          <button
            onClick={async () => {
              try {
                await window.aistudio?.openSelectKey();
                setHasKey(true);
              } catch (e) {
                console.error(e);
              }
            }}
            className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-3 px-4 rounded-xl transition-colors"
          >
            Select API Key
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50 font-sans selection:bg-indigo-500/30 pb-20">
      {/* Header */}
      <header className="sticky top-0 z-50 glass-panel border-b border-zinc-800/50 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-indigo-500 flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <h1 className="text-xl font-semibold tracking-tight hidden sm:block">FaceSwap Studio</h1>
        </div>
        
        <div className="flex bg-zinc-900 rounded-full p-1 border border-zinc-800 overflow-x-auto hide-scrollbar">
          <button
            onClick={() => setActiveTab('profile')}
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors flex items-center gap-2 whitespace-nowrap ${activeTab === 'profile' ? 'bg-zinc-800 text-white shadow-sm' : 'text-zinc-400 hover:text-zinc-200'}`}
          >
            <User className="w-4 h-4" />
            Identity
          </button>
          <button
            onClick={() => setActiveTab('studio')}
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors flex items-center gap-2 whitespace-nowrap ${activeTab === 'studio' ? 'bg-zinc-800 text-white shadow-sm' : 'text-zinc-400 hover:text-zinc-200'}`}
          >
            <ImageIcon className="w-4 h-4" />
            Studio
          </button>
          <button
            onClick={() => setActiveTab('style')}
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors flex items-center gap-2 whitespace-nowrap ${activeTab === 'style' ? 'bg-zinc-800 text-white shadow-sm' : 'text-zinc-400 hover:text-zinc-200'}`}
          >
            <Palette className="w-4 h-4" />
            Style Extractor
          </button>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 mt-8">
        {error && (
          <div className="mb-8 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 flex items-start gap-3">
            <Info className="w-5 h-5 shrink-0 mt-0.5" />
            <p>{error}</p>
          </div>
        )}

        <AnimatePresence mode="wait">
          {activeTab === 'profile' ? (
            <motion.div
              key="profile"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="max-w-3xl mx-auto"
            >
              <div className="text-center mb-10">
                <h2 className="text-3xl font-semibold tracking-tight mb-3">Your Identity</h2>
                <p className="text-zinc-400">Upload clear photos of your face from different angles. These will be used to accurately recreate your likeness in the studio.</p>
              </div>

              <FileDropzone onDrop={handleSelfImagesDrop} multiple className="h-48 mb-8">
                <div className="w-12 h-12 rounded-full bg-zinc-800 flex items-center justify-center mb-4">
                  <Upload className="w-6 h-6 text-zinc-400" />
                </div>
                <p className="font-medium mb-1">Click or drag photos here (Select multiple)</p>
                <p className="text-sm text-zinc-500">Supports multiple images (JPG, PNG, HEIC)</p>
              </FileDropzone>

              {selfImages.length > 0 && (
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="font-medium text-lg">Reference Photos ({selfImages.length}/10)</h3>
                    <button onClick={() => setSelfImages([])} className="text-sm text-zinc-500 hover:text-red-400 transition-colors">Clear All</button>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                    <AnimatePresence>
                      {selfImages.map((img, idx) => (
                        <motion.div
                          key={idx}
                          initial={{ opacity: 0, scale: 0.9 }}
                          animate={{ opacity: 1, scale: 1 }}
                          exit={{ opacity: 0, scale: 0.9 }}
                          className={`relative aspect-square rounded-xl overflow-hidden group border ${bestMatchIndex === idx ? 'border-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.3)]' : 'border-zinc-800'}`}
                        >
                          <img src={img} alt={`Reference ${idx}`} className="w-full h-full object-cover" />
                          {bestMatchIndex === idx && (
                            <div className="absolute top-2 left-2 bg-emerald-500 text-white text-[10px] font-bold px-2 py-1 rounded-full flex items-center gap-1">
                              <Check className="w-3 h-3" /> Best Match
                            </div>
                          )}
                          <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                            <button
                              onClick={() => removeSelfImage(idx)}
                              className="p-2 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>
                        </motion.div>
                      ))}
                    </AnimatePresence>
                  </div>
                </div>
              )}
            </motion.div>
          ) : activeTab === 'studio' ? (
            <motion.div
              key="studio"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="grid lg:grid-cols-[400px_1fr] gap-8"
            >
              {/* Left Column: Controls */}
              <div className="space-y-6">
                <div className="glass-panel p-6 rounded-2xl">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="font-medium text-lg">Target Photo</h3>
                    {targetImage && (
                      <button 
                        onClick={() => setShowWireframe(!showWireframe)}
                        className={`p-1.5 rounded-lg transition-colors ${showWireframe ? 'bg-indigo-500/20 text-indigo-400' : 'bg-zinc-800 text-zinc-400 hover:text-white'}`}
                        title="Toggle Face Wireframe"
                      >
                        <ScanFace className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                  
                  {targetImage ? (
                    <div className="space-y-3 mb-4">
                      <div className="relative w-full rounded-xl overflow-hidden border border-zinc-800 bg-black">
                        <img 
                          ref={targetImageRef}
                          src={targetImage.url} 
                          alt="Target" 
                          className="w-full h-auto block cursor-crosshair opacity-80 hover:opacity-100 transition-opacity" 
                          onClick={handleImageClick}
                          onLoad={detectAndDrawLandmarks}
                        />
                        <canvas 
                          ref={canvasRef}
                          className={`absolute top-0 left-0 w-full h-full pointer-events-none transition-opacity duration-300 ${showWireframe ? 'opacity-100' : 'opacity-0'}`}
                        />
                        {faceRect && targetImgDimensions && (
                          <div
                            className="absolute border-2 border-emerald-500 bg-emerald-500/20 pointer-events-none transition-all duration-300"
                            style={{
                              left: `${(faceRect.x / targetImgDimensions.width) * 100}%`,
                              top: `${(faceRect.y / targetImgDimensions.height) * 100}%`,
                              width: `${(faceRect.width / targetImgDimensions.width) * 100}%`,
                              height: `${(faceRect.height / targetImgDimensions.height) * 100}%`,
                              boxShadow: '0 0 0 9999px rgba(0,0,0,0.4)'
                            }}
                          >
                            <div className="absolute -top-6 left-0 bg-emerald-500 text-white text-xs px-2 py-0.5 rounded-t-md font-medium whitespace-nowrap flex items-center gap-1">
                              <Focus className="w-3 h-3" /> Target Face
                            </div>
                          </div>
                        )}
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <p className="text-zinc-400 flex items-center gap-1.5">
                          <Info className="w-4 h-4" /> Click image to adjust face target
                        </p>
                        <button 
                          onClick={() => { setTargetImage(null); setFaceRect(null); setCroppedFaceUrl(null); }}
                          className="text-indigo-400 hover:text-indigo-300 font-medium"
                        >
                          Replace Photo
                        </button>
                      </div>
                    </div>
                  ) : (
                    <FileDropzone onDrop={handleTargetImageDrop} className="aspect-[3/4] mb-4">
                      <div className="flex flex-col items-center text-center p-6">
                        <div className="w-12 h-12 rounded-full bg-zinc-800 flex items-center justify-center mb-4">
                          <ImageIcon className="w-6 h-6 text-zinc-400" />
                        </div>
                        <p className="font-medium mb-1">Upload Target</p>
                        <p className="text-sm text-zinc-500">The photo you want to be swapped into</p>
                      </div>
                    </FileDropzone>
                  )}

                  <div className="space-y-4 mt-6 border-t border-zinc-800/50 pt-6">
                    <h4 className="text-sm font-medium text-zinc-300 flex items-center gap-2">
                      <Sliders className="w-4 h-4 text-indigo-400" /> Face Adjustments
                    </h4>
                    
                    <div className="space-y-4">
                      <div>
                        <div className="flex justify-between text-xs text-zinc-400 mb-1.5">
                          <span>Head Scale</span>
                          <span className="text-indigo-400">{faceAdjustments.scale}%</span>
                        </div>
                        <input type="range" min="80" max="120" value={faceAdjustments.scale} onChange={(e) => setFaceAdjustments({...faceAdjustments, scale: parseInt(e.target.value)})} className="w-full accent-indigo-500" />
                      </div>
                      <div>
                        <div className="flex justify-between text-xs text-zinc-400 mb-1.5">
                          <span>Jawline</span>
                          <span className="text-indigo-400">{faceAdjustments.jawline > 0 ? `+${faceAdjustments.jawline}` : faceAdjustments.jawline}</span>
                        </div>
                        <input type="range" min="-50" max="50" value={faceAdjustments.jawline} onChange={(e) => setFaceAdjustments({...faceAdjustments, jawline: parseInt(e.target.value)})} className="w-full accent-indigo-500" />
                      </div>
                      <div>
                        <div className="flex justify-between text-xs text-zinc-400 mb-1.5">
                          <span>Nose Size</span>
                          <span className="text-indigo-400">{faceAdjustments.nose > 0 ? `+${faceAdjustments.nose}` : faceAdjustments.nose}</span>
                        </div>
                        <input type="range" min="-50" max="50" value={faceAdjustments.nose} onChange={(e) => setFaceAdjustments({...faceAdjustments, nose: parseInt(e.target.value)})} className="w-full accent-indigo-500" />
                      </div>
                      <div>
                        <div className="flex justify-between text-xs text-zinc-400 mb-1.5">
                          <span>Eye Size</span>
                          <span className="text-indigo-400">{faceAdjustments.eyes > 0 ? `+${faceAdjustments.eyes}` : faceAdjustments.eyes}</span>
                        </div>
                        <input type="range" min="-50" max="50" value={faceAdjustments.eyes} onChange={(e) => setFaceAdjustments({...faceAdjustments, eyes: parseInt(e.target.value)})} className="w-full accent-indigo-500" />
                      </div>
                    </div>
                  </div>

                  <div className="space-y-2 mt-6 border-t border-zinc-800/50 pt-6">
                    <label className="text-sm font-medium text-zinc-400">Advanced Instructions (Optional)</label>
                    <textarea
                      value={instructions}
                      onChange={(e) => setInstructions(e.target.value)}
                      placeholder="e.g., Make my expression slightly smiling, pose me looking left, match the lighting from the left side..."
                      className="w-full bg-zinc-900 border border-zinc-800 rounded-xl p-3 text-sm focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all resize-none h-24"
                    />
                  </div>

                  <button
                    onClick={handleGenerate}
                    disabled={isGenerating || !targetImage}
                    className="w-full mt-6 bg-indigo-600 hover:bg-indigo-700 disabled:bg-zinc-800 disabled:text-zinc-500 text-white font-medium py-3 px-4 rounded-xl transition-colors flex items-center justify-center gap-2"
                  >
                    {isGenerating ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        {isAnalyzingPose ? "Finding Best Match..." : "Generating..."}
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-5 h-5" />
                        Generate Swap
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Right Column: Results */}
              <div className="glass-panel p-6 rounded-2xl min-h-[600px] flex flex-col">
                <h3 className="font-medium text-lg mb-6 flex items-center gap-2">
                  Results
                  {results.length > 0 && <span className="text-xs bg-zinc-800 px-2 py-1 rounded-full text-zinc-400">{results.length} generated</span>}
                </h3>

                {isGenerating ? (
                  <div className="flex-1 flex flex-col items-center justify-center text-zinc-500 space-y-4">
                    <Loader2 className="w-10 h-10 animate-spin text-indigo-500" />
                    <p>{isAnalyzingPose ? "Analyzing 3D head pose and selecting best reference..." : "Applying structural adjustments and generating swap..."}</p>
                  </div>
                ) : results.length > 0 ? (
                  <div className="flex flex-col h-full">
                    <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 flex-1">
                      {results.map((img, idx) => (
                        <motion.div
                          key={idx}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: idx * 0.1 }}
                          className="group"
                        >
                          <div className="relative aspect-[3/4] rounded-xl overflow-hidden border border-zinc-800 mb-3">
                            <img src={img} alt={`Result ${idx}`} className="w-full h-full object-cover" />
                          </div>
                          <div className="flex items-center justify-center gap-2">
                            <button
                              onClick={() => handleThumbsUp(img)}
                              className={`flex-1 py-2 px-3 rounded-lg flex items-center justify-center gap-2 text-sm font-medium transition-colors ${likedImages.includes(img) ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-white hover:bg-zinc-800'}`}
                            >
                              <ThumbsUp className="w-4 h-4" />
                              {likedImages.includes(img) ? 'Liked' : 'Like'}
                            </button>
                            <button
                              onClick={() => handleThumbsDown(img)}
                              className="flex-1 py-2 px-3 rounded-lg flex items-center justify-center gap-2 text-sm font-medium bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-red-400 hover:bg-red-500/10 hover:border-red-500/30 transition-colors"
                            >
                              <ThumbsDown className="w-4 h-4" />
                              Dislike
                            </button>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                    
                    {recreationPrompt && (
                      <motion.div 
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="mt-8 pt-8 border-t border-zinc-800/50"
                      >
                        <h4 className="font-medium text-sm text-zinc-400 mb-3 flex items-center gap-2">
                          <Sparkles className="w-4 h-4" /> Recreation Prompt
                        </h4>
                        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 relative group">
                          <p className="text-sm text-zinc-300 leading-relaxed pr-10">{recreationPrompt}</p>
                          <button
                            onClick={() => {
                              navigator.clipboard.writeText(recreationPrompt);
                              setCopiedRecreation(true);
                              setTimeout(() => setCopiedRecreation(false), 2000);
                            }}
                            className="absolute top-4 right-4 p-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-zinc-400 hover:text-white transition-colors"
                            title="Copy prompt"
                          >
                            {copiedRecreation ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                          </button>
                        </div>
                        <p className="text-xs text-zinc-500 mt-2">
                          Use this prompt in any AI image generator to recreate this exact scene and style.
                        </p>
                      </motion.div>
                    )}
                  </div>
                ) : (
                  <div className="flex-1 flex flex-col items-center justify-center text-zinc-500 text-center max-w-sm mx-auto">
                    <div className="w-16 h-16 rounded-full bg-zinc-900 flex items-center justify-center mb-4 border border-zinc-800">
                      <ImageIcon className="w-8 h-8 text-zinc-600" />
                    </div>
                    <p className="font-medium text-zinc-300 mb-2">Ready to create</p>
                    <p className="text-sm">Upload a target photo and click generate to see yourself in a new style.</p>
                  </div>
                )}
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="style"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="grid lg:grid-cols-[400px_1fr] gap-8"
            >
              {/* Left Column: Style Upload */}
              <div className="space-y-6">
                <div className="glass-panel p-6 rounded-2xl">
                  <h3 className="font-medium text-lg mb-4">Style Reference</h3>
                  <FileDropzone onDrop={(files) => {
                    if (files.length > 0) {
                      const reader = new FileReader();
                      reader.onloadend = () => setStyleImage(reader.result as string);
                      reader.readAsDataURL(files[0]);
                    }
                  }} className="aspect-[3/4] mb-4">
                    {styleImage ? (
                      <div className="relative w-full h-full group">
                        <img src={styleImage} alt="Style" className="w-full h-full object-cover rounded-xl" />
                        <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center rounded-xl">
                          <p className="text-white font-medium flex items-center gap-2">
                            <Upload className="w-4 h-4" /> Replace Photo
                          </p>
                        </div>
                      </div>
                    ) : (
                      <div className="flex flex-col items-center text-center p-6">
                        <div className="w-12 h-12 rounded-full bg-zinc-800 flex items-center justify-center mb-4">
                          <Palette className="w-6 h-6 text-zinc-400" />
                        </div>
                        <p className="font-medium mb-1">Upload Style Photo</p>
                        <p className="text-sm text-zinc-500">Upload an image to extract its style prompt</p>
                      </div>
                    )}
                  </FileDropzone>
                  <button
                    onClick={handleAnalyzeStyle}
                    disabled={isAnalyzingStyle || !styleImage}
                    className="w-full mt-6 bg-indigo-600 hover:bg-indigo-700 disabled:bg-zinc-800 disabled:text-zinc-500 text-white font-medium py-3 px-4 rounded-xl transition-colors flex items-center justify-center gap-2"
                  >
                    {isAnalyzingStyle ? (
                      <><Loader2 className="w-5 h-5 animate-spin" /> Analyzing...</>
                    ) : (
                      <><Sparkles className="w-5 h-5" /> Extract Prompt</>
                    )}
                  </button>
                </div>
              </div>

              {/* Right Column: Extracted Prompt */}
              <div className="glass-panel p-6 rounded-2xl min-h-[600px] flex flex-col">
                <h3 className="font-medium text-lg mb-6 flex items-center gap-2">Extracted Prompt</h3>
                {isAnalyzingStyle ? (
                  <div className="flex-1 flex flex-col items-center justify-center text-zinc-500 space-y-4">
                    <Loader2 className="w-10 h-10 animate-spin text-indigo-500" />
                    <p>Analyzing style, lighting, and composition...</p>
                  </div>
                ) : stylePrompt ? (
                  <div className="flex-1 flex flex-col">
                    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 flex-1 relative group">
                      <p className="text-zinc-300 leading-relaxed whitespace-pre-wrap">{stylePrompt}</p>
                      <button
                        onClick={() => {
                          navigator.clipboard.writeText(stylePrompt);
                          setCopiedStyle(true);
                          setTimeout(() => setCopiedStyle(false), 2000);
                        }}
                        className="absolute top-4 right-4 p-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-zinc-400 hover:text-white transition-colors"
                        title="Copy prompt"
                      >
                        {copiedStyle ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                      </button>
                    </div>
                    <p className="text-sm text-zinc-500 mt-4">
                      You can use this prompt in any AI image generator to recreate this style, or paste it into the Studio tab's instructions to guide the face swap.
                    </p>
                  </div>
                ) : (
                  <div className="flex-1 flex flex-col items-center justify-center text-zinc-500 text-center max-w-sm mx-auto">
                    <div className="w-16 h-16 rounded-full bg-zinc-900 flex items-center justify-center mb-4 border border-zinc-800">
                      <Palette className="w-8 h-8 text-zinc-600" />
                    </div>
                    <p className="font-medium text-zinc-300 mb-2">Extract Style Prompts</p>
                    <p className="text-sm">Upload a photo you love the look of, and we'll generate a detailed prompt you can use to recreate its exact aesthetic.</p>
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
