import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as faceapi from '@vladmandic/face-api';
import { Camera, CameraOff, Users, TrendingUp, Download, Settings, Save, AlertCircle, Activity, Dog, UserPlus, ChevronDown, ChevronUp, List, X, Trash2 } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

// Utility for tailwind class merging
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface Track {
  id: number;
  centroid: { x: number; y: number };
  bbox: number[];
  lastSeen: number;
  hitCount: number;
  counted: boolean;
}

interface DetectionLogEntry {
  id: string;
  type: 'person' | 'pet';
  timestamp: string;
  bbox: number[];
  score: number;
}

interface HeatmapZone {
  id: string;
  name: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

interface DailyStat {
  date: string;
  peakCount: number;
  totalCount: number;
  totalPetCount: number;
}

// Toast Component
const Toast = ({ message, type, onClose }: { message: string, type: 'error' | 'success', onClose: () => void }) => {
  useEffect(() => {
    const timer = setTimeout(onClose, 5000);
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div className={cn(
      "fixed bottom-4 right-4 flex items-center gap-2 px-4 py-3 rounded-lg shadow-lg text-white z-50 animate-in slide-in-from-bottom-5",
      type === 'error' ? "bg-red-600" : "bg-emerald-600"
    )}>
      {type === 'error' ? <AlertCircle className="w-5 h-5" /> : <Save className="w-5 h-5" />}
      <span className="text-sm font-medium">{message}</span>
    </div>
  );
};

export default function App() {
  // State
  const [isSystemActive, setIsSystemActive] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [isFaceModelLoading, setIsFaceModelLoading] = useState(true);
  const [currentCount, setCurrentCount] = useState(0);
  const [peakCount, setPeakCount] = useState(0);
  const [history, setHistory] = useState<{ timestamp: number; count: number; petCount?: number; zoneScores?: Record<string, number> }[]>([]);
  const [dataApiUrl, setDataApiUrl] = useState('');
  const [imageApiUrl, setImageApiUrl] = useState('');
  const [toast, setToast] = useState<{ message: string; type: 'error' | 'success' } | null>(null);
  const [isPetCountingEnabled, setIsPetCountingEnabled] = useState(false);
  const [currentPetCount, setCurrentPetCount] = useState(0);
  const [uniquePersonsToday, setUniquePersonsToday] = useState(0);
  const [totalPetCountToday, setTotalPetCountToday] = useState(0);
  
  // New settings
  const [personConfidence, setPersonConfidence] = useState(0.6);
  const [petConfidence, setPetConfidence] = useState(0.7);
  const [faceMatchThreshold, setFaceMatchThreshold] = useState(0.55);
  const [isFaceBlurEnabled, setIsFaceBlurEnabled] = useState(true);
  
  // Detection Log
  const [detectionLog, setDetectionLog] = useState<DetectionLogEntry[]>([]);
  const [isLogModalOpen, setIsLogModalOpen] = useState(false);
  const detectionLogRef = useRef<DetectionLogEntry[]>([]);
  const [availableCameras, setAvailableCameras] = useState<MediaDeviceInfo[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string>('');
  const [storeCode, setStoreCode] = useState('');
  const [detectionIntervalMs, setDetectionIntervalMs] = useState(1000);
  const [dailyStats, setDailyStats] = useState<DailyStat[]>([]);
  const [isSettingsOpen, setIsSettingsOpen] = useState(true);
  const [isAdvancedSettingsOpen, setIsAdvancedSettingsOpen] = useState(false);
  const [isRecordingHeatmap, setIsRecordingHeatmap] = useState(false);
  const [zones, setZones] = useState<HeatmapZone[]>([]);
  const [isEditingZones, setIsEditingZones] = useState(false);
  const [drawingZone, setDrawingZone] = useState<Partial<HeatmapZone> | null>(null);

  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const heatmapCanvasRef = useRef<HTMLCanvasElement>(null);
  const modelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const detectionIntervalRef = useRef<number | null>(null);
  const webhookIntervalRef = useRef<number | null>(null);
  const recentCountsRef = useRef<{ timestamp: number; count: number }[]>([]);
  const recentPetCountsRef = useRef<{ timestamp: number; count: number }[]>([]);
  const tracksRef = useRef<Track[]>([]);
  const nextTrackIdRef = useRef<number>(1);
  const knownFaceDescriptorsRef = useRef<Float32Array[]>([]);

  // Refs for intervals to avoid stale closures
  const currentCountRef = useRef(currentCount);
  const peakCountRef = useRef(peakCount);
  const dataApiUrlRef = useRef(dataApiUrl);
  const imageApiUrlRef = useRef(imageApiUrl);
  const isSystemActiveRef = useRef(isSystemActive);
  const isPetCountingEnabledRef = useRef(isPetCountingEnabled);
  const currentPetCountRef = useRef(currentPetCount);
  const uniquePersonsTodayRef = useRef(uniquePersonsToday);
  const totalPetCountTodayRef = useRef(totalPetCountToday);
  const personConfidenceRef = useRef(personConfidence);
  const petConfidenceRef = useRef(petConfidence);
  const faceMatchThresholdRef = useRef(faceMatchThreshold);
  const isFaceBlurEnabledRef = useRef(isFaceBlurEnabled);
  const zonesRef = useRef<HeatmapZone[]>([]);
  const isEditingZonesRef = useRef(false);
  const drawingZoneRef = useRef<Partial<HeatmapZone> | null>(null);
  const zoneHitsRef = useRef<Record<string, number>>({});
  const detectionIntervalMsRef = useRef(detectionIntervalMs);
  const storeCodeRef = useRef(storeCode);
  const isRecordingHeatmapRef = useRef(isRecordingHeatmap);

  useEffect(() => { currentCountRef.current = currentCount; }, [currentCount]);
  useEffect(() => { peakCountRef.current = peakCount; }, [peakCount]);
  useEffect(() => { dataApiUrlRef.current = dataApiUrl; }, [dataApiUrl]);
  useEffect(() => { imageApiUrlRef.current = imageApiUrl; }, [imageApiUrl]);
  useEffect(() => { isSystemActiveRef.current = isSystemActive; }, [isSystemActive]);
  useEffect(() => { isPetCountingEnabledRef.current = isPetCountingEnabled; }, [isPetCountingEnabled]);
  useEffect(() => { currentPetCountRef.current = currentPetCount; }, [currentPetCount]);
  useEffect(() => { uniquePersonsTodayRef.current = uniquePersonsToday; }, [uniquePersonsToday]);
  useEffect(() => { totalPetCountTodayRef.current = totalPetCountToday; }, [totalPetCountToday]);
  useEffect(() => { personConfidenceRef.current = personConfidence; }, [personConfidence]);
  useEffect(() => { petConfidenceRef.current = petConfidence; }, [petConfidence]);
  useEffect(() => { faceMatchThresholdRef.current = faceMatchThreshold; }, [faceMatchThreshold]);
  useEffect(() => { isFaceBlurEnabledRef.current = isFaceBlurEnabled; }, [isFaceBlurEnabled]);
  useEffect(() => { zonesRef.current = zones; }, [zones]);
  useEffect(() => { isEditingZonesRef.current = isEditingZones; }, [isEditingZones]);
  useEffect(() => { drawingZoneRef.current = drawingZone; }, [drawingZone]);
  useEffect(() => { detectionIntervalMsRef.current = detectionIntervalMs; }, [detectionIntervalMs]);
  useEffect(() => { storeCodeRef.current = storeCode; }, [storeCode]);
  useEffect(() => { isRecordingHeatmapRef.current = isRecordingHeatmap; }, [isRecordingHeatmap]);

  // Load Model
  useEffect(() => {
    const loadModel = async () => {
      try {
        await tf.ready();
        const model = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
        modelRef.current = model;
        setIsModelLoading(false);

        // Load face-api models
        const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/';
        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
        await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
        await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
        setIsFaceModelLoading(false);
      } catch (err) {
        console.error("Failed to load model:", err);
        setToast({ message: 'AI 模型載入失敗，請重整頁面。', type: 'error' });
      }
    };
    loadModel();
    
    // Load history from local storage
    const savedHistory = localStorage.getItem('peopleCountHistory');
    if (savedHistory) {
      try {
        setHistory(JSON.parse(savedHistory));
      } catch (e) {}
    }
    
    // Load webhook URL from local storage
    const savedDataApi = localStorage.getItem('dataApiUrl');
    if (savedDataApi) {
      setDataApiUrl(savedDataApi);
    }
    const savedImageApi = localStorage.getItem('imageApiUrl');
    if (savedImageApi) {
      setImageApiUrl(savedImageApi);
    }

    // Load store code
    const savedStoreCode = localStorage.getItem('storeCode');
    if (savedStoreCode) {
      setStoreCode(savedStoreCode);
    }

    // Load daily stats
    const savedDailyStats = localStorage.getItem('dailyStats');
    if (savedDailyStats) {
      try {
        setDailyStats(JSON.parse(savedDailyStats));
      } catch (e) {}
    }

    // Load pet counting setting
    const savedPetCounting = localStorage.getItem('isPetCountingEnabled');
    if (savedPetCounting) {
      setIsPetCountingEnabled(savedPetCounting === 'true');
    }

    const savedPersonConf = localStorage.getItem('personConfidence');
    if (savedPersonConf) setPersonConfidence(parseFloat(savedPersonConf));

    const savedPetConf = localStorage.getItem('petConfidence');
    if (savedPetConf) setPetConfidence(parseFloat(savedPetConf));

    const savedFaceMatch = localStorage.getItem('faceMatchThreshold');
    if (savedFaceMatch) setFaceMatchThreshold(parseFloat(savedFaceMatch));

    const savedFaceBlur = localStorage.getItem('isFaceBlurEnabled');
    if (savedFaceBlur) setIsFaceBlurEnabled(savedFaceBlur === 'true');

    // Load detection interval
    const savedInterval = localStorage.getItem('detectionIntervalMs');
    if (savedInterval) {
      setDetectionIntervalMs(parseInt(savedInterval, 10));
    }

    // Load total count today
    const savedTotal = localStorage.getItem('uniquePersonsToday');
    const savedPetTotal = localStorage.getItem('totalPetCountToday');
    const savedDate = localStorage.getItem('totalCountDate');
    const today = new Date().toDateString();
    
    if (savedTotal && savedDate === today) {
      setUniquePersonsToday(parseInt(savedTotal, 10));
    } else {
      setUniquePersonsToday(0);
      localStorage.setItem('uniquePersonsToday', '0');
      localStorage.setItem('totalCountDate', today);
    }

    if (savedPetTotal && savedDate === today) {
      setTotalPetCountToday(parseInt(savedPetTotal, 10));
    } else {
      setTotalPetCountToday(0);
      localStorage.setItem('totalPetCountToday', '0');
    }

    // Get available cameras
    const getCameras = async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        setAvailableCameras(videoDevices);
        if (videoDevices.length > 0 && videoDevices[0].deviceId) {
          setSelectedCameraId(videoDevices[0].deviceId);
        }
      } catch (err) {
        console.error("Error getting cameras:", err);
      }
    };
    getCameras();
  }, []);

  // Save history to local storage
  useEffect(() => {
    localStorage.setItem('peopleCountHistory', JSON.stringify(history.slice(-1000))); // Keep last 1000 records
  }, [history]);

  // Update daily stats
  useEffect(() => {
    const today = new Date().toISOString().split('T')[0];
    setDailyStats(prev => {
      const existingIndex = prev.findIndex(s => s.date === today);
      const newStat = {
        date: today,
        peakCount,
        totalCount: uniquePersonsToday,
        totalPetCount: totalPetCountToday
      };
      
      let nextStats;
      if (existingIndex >= 0) {
        nextStats = [...prev];
        nextStats[existingIndex] = newStat;
      } else {
        nextStats = [...prev, newStat];
      }
      
      // Keep only last 30 days
      if (nextStats.length > 30) {
        nextStats = nextStats.slice(-30);
      }
      
      localStorage.setItem('dailyStats', JSON.stringify(nextStats));
      return nextStats;
    });
  }, [peakCount, uniquePersonsToday, totalPetCountToday]);

  // Daily reset for peak count and total count
  useEffect(() => {
    const checkMidnight = setInterval(() => {
      const now = new Date();
      if (now.getHours() === 0 && now.getMinutes() === 0 && now.getSeconds() === 0) {
        setPeakCount(0);
        setUniquePersonsToday(0);
        setTotalPetCountToday(0);
        localStorage.setItem('uniquePersonsToday', '0');
        localStorage.setItem('totalPetCountToday', '0');
        localStorage.setItem('totalCountDate', now.toDateString());
        tracksRef.current = [];
        knownFaceDescriptorsRef.current = [];
      }
    }, 1000);
    return () => clearInterval(checkMidnight);
  }, []);

  // Start/Stop Camera
  const toggleSystem = async () => {
    if (isSystemActive) {
      // Stop System
      stopSystem();
    } else {
      // Start System
      await startSystem();
    }
  };

  const switchCamera = async (deviceId: string) => {
    setSelectedCameraId(deviceId);
    if (isSystemActive) {
      // Stop current stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      // Restart with new camera
      await startSystem(deviceId);
    }
  };

  const startSystem = async (specificDeviceId?: string) => {
    if (!storeCode.trim()) {
      setToast({ message: '請先輸入店櫃代號才可以啟動系統。', type: 'error' });
      return;
    }

    if (!modelRef.current || isFaceModelLoading) {
      setToast({ message: 'AI 模型尚未載入完成，請稍候。', type: 'error' });
      return;
    }

    try {
      let videoConstraints: MediaTrackConstraints = { width: { ideal: 1280 }, height: { ideal: 720 } };
      
      const targetDeviceId = specificDeviceId || selectedCameraId;
      if (targetDeviceId) {
        videoConstraints.deviceId = { exact: targetDeviceId };
      } else {
        videoConstraints.facingMode = 'environment';
      }

      const constraints: MediaStreamConstraints = {
        video: videoConstraints,
        audio: false
      };
      
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
      } catch (initialErr: any) {
        console.warn("Initial camera request failed, trying fallback...", initialErr);
        // Fallback to the most basic video request if constraints fail
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      }
      
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play().catch(e => console.error("Video play error:", e));
          setIsSystemActive(true);
          startDetection();
          startWebhookInterval();
        };
      }

      // After permission is granted, re-enumerate devices to get labels
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');
      setAvailableCameras(videoDevices);
      if (!selectedCameraId && videoDevices.length > 0) {
        const activeTrack = stream.getVideoTracks()[0];
        const activeDevice = videoDevices.find(d => d.label === activeTrack.label);
        if (activeDevice) {
          setSelectedCameraId(activeDevice.deviceId);
        } else {
          setSelectedCameraId(videoDevices[0].deviceId);
        }
      }
    } catch (err: any) {
      console.error("Camera error:", err);
      let errorMessage = '無法存取攝影機，請確認權限設定。';
      if (err.name === 'NotAllowedError' || err.message === 'Permission denied') {
        errorMessage = '攝影機權限被拒絕，請在瀏覽器網址列解鎖攝影機權限後重試。';
      } else if (err.name === 'NotFoundError') {
        errorMessage = '找不到攝影機設備，請確認是否有連接攝影機。';
      }
      setToast({ message: errorMessage, type: 'error' });
    }
  };

  const handleCameraChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newCameraId = e.target.value;
    setSelectedCameraId(newCameraId);
    
    if (isSystemActive) {
      // Stop current stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      
      // Start new stream with selected camera
      try {
        let stream: MediaStream;
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: { deviceId: { exact: newCameraId }, width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false
          });
        } catch (initialErr) {
          console.warn("Camera switch constraint failed, trying fallback...", initialErr);
          stream = await navigator.mediaDevices.getUserMedia({
            video: { deviceId: { exact: newCameraId } },
            audio: false
          });
        }
        
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play().catch(e => console.error("Video play error:", e));
          };
        }
      } catch (err: any) {
        console.error("Camera switch error:", err);
        let errorMessage = '切換攝影機失敗。';
        if (err.name === 'NotAllowedError' || err.message === 'Permission denied') {
          errorMessage = '攝影機權限被拒絕。';
        } else if (err.name === 'NotFoundError') {
          errorMessage = '找不到該攝影機設備。';
        }
        setToast({ message: errorMessage, type: 'error' });
      }
    }
  };

  const stopSystem = () => {
    // Stop video tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    
    // Stop detection loop
    if (detectionIntervalRef.current) {
      window.clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    
    // Stop webhook interval
    if (webhookIntervalRef.current) {
      window.clearInterval(webhookIntervalRef.current);
      webhookIntervalRef.current = null;
    }
    
    // Clear canvas
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
    
    setIsSystemActive(false);
    setCurrentCount(0);
    setCurrentPetCount(0);
    recentCountsRef.current = [];
    recentPetCountsRef.current = [];
    // Note: We do NOT clear tracksRef.current here to persist data across camera switches
  };

  // Detection Loop
  const startDetection = () => {
    if (detectionIntervalRef.current) {
      window.clearInterval(detectionIntervalRef.current);
    }

    detectionIntervalRef.current = window.setInterval(async () => {
      if (!videoRef.current || !canvasRef.current || !modelRef.current || !heatmapCanvasRef.current) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const heatmapCanvas = heatmapCanvasRef.current;
      
      if (video.readyState < 2) return; // Ensure video has enough data

      // Ensure canvas dimensions match video
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
      if (heatmapCanvas.width !== video.videoWidth || heatmapCanvas.height !== video.videoHeight) {
        heatmapCanvas.width = video.videoWidth;
        heatmapCanvas.height = video.videoHeight;
      }

      try {
        const predictions = await modelRef.current.detect(video);
        if (!isSystemActiveRef.current) return; // Prevent updates if stopped during detection
        
        const people = predictions.filter(p => p.class === 'person' && p.score > personConfidenceRef.current);
        const pets = isPetCountingEnabledRef.current 
          ? predictions.filter(p => (p.class === 'cat' || p.class === 'dog') && p.score > petConfidenceRef.current) 
          : [];
        
        const trackedPeople = updateTracking(people);
        const trackedPets = updatePetTracking(pets);
        drawBoundingBoxes(trackedPeople, trackedPets, canvas, isPetCountingEnabledRef.current, video);
        updateCount(people.length, pets.length);

        // Heatmap Drawing Logic
        if (isRecordingHeatmapRef.current) {
          const heatmapCtx = heatmapCanvas.getContext('2d');
          if (heatmapCtx) {
            people.forEach(person => {
              const [x, y, width, height] = person.bbox;
              const bottomCenterX = x + width / 2;
              const bottomCenterY = y + height;
              
              const radius = Math.max(width, height) / 2;
              const gradient = heatmapCtx.createRadialGradient(
                bottomCenterX, bottomCenterY, 0,
                bottomCenterX, bottomCenterY, radius
              );
              
              gradient.addColorStop(0, 'rgba(255, 69, 0, 0.05)');
              gradient.addColorStop(1, 'rgba(255, 69, 0, 0)');
              
              heatmapCtx.fillStyle = gradient;
              heatmapCtx.beginPath();
              heatmapCtx.arc(bottomCenterX, bottomCenterY, radius, 0, 2 * Math.PI);
              heatmapCtx.fill();
            });
          }
        }
        
        // Face recognition for unique count
        processFaces(people, video);
      } catch (err) {
        console.error("Detection error:", err);
      }
    }, detectionIntervalMsRef.current);
  };

  // Restart detection loop when interval changes
  useEffect(() => {
    if (isSystemActive) {
      startDetection();
    }
  }, [detectionIntervalMs]);

  const processFaces = async (people: cocoSsd.DetectedObject[], video: HTMLVideoElement) => {
    for (const person of people) {
      const [x, y, width, height] = person.bbox;
      
      // Skip if bounding box is too small or out of bounds
      if (width < 30 || height < 30 || x < 0 || y < 0 || x + width > video.videoWidth || y + height > video.videoHeight) {
        continue;
      }

      const cropCanvas = document.createElement('canvas');
      cropCanvas.width = width;
      cropCanvas.height = height;
      const cropCtx = cropCanvas.getContext('2d');
      if (!cropCtx) continue;

      cropCtx.drawImage(video, x, y, width, height, 0, 0, width, height);

      try {
        const face = await faceapi.detectSingleFace(cropCanvas, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptor();
        
        if (face) {
          const descriptor = face.descriptor;
          let isKnown = false;
          for (const known of knownFaceDescriptorsRef.current) {
            const distance = faceapi.euclideanDistance(descriptor, known);
            if (distance <= faceMatchThresholdRef.current) {
              isKnown = true;
              break;
            }
          }
          
          if (!isKnown) {
            knownFaceDescriptorsRef.current.push(descriptor);
            setUniquePersonsToday(prev => {
              const next = prev + 1;
              localStorage.setItem('uniquePersonsToday', String(next));
              return next;
            });
          }
        }
      } catch (err) {
        // Ignore face detection errors for individual crops
      }
    }
  };

  const drawBoundingBoxes = (trackedPeople: { detection: cocoSsd.DetectedObject, trackId: number }[], trackedPets: { detection: cocoSsd.DetectedObject, trackId: number }[], canvas: HTMLCanvasElement, petEnabled: boolean, video: HTMLVideoElement) => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const drawBox = (prediction: cocoSsd.DetectedObject, colorHex: string, colorRgb: string, labelPrefix: string) => {
      const [x, y, width, height] = prediction.bbox;

      if (isFaceBlurEnabledRef.current && prediction.class === 'person') {
        const faceHeight = height * 0.4; // 增加遮蔽範圍以確保覆蓋到眼睛
        
        ctx.save();
        ctx.beginPath();
        ctx.rect(x, y, width, faceHeight);
        ctx.clip();
        
        ctx.filter = 'blur(15px)';
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        ctx.restore();
      }

      // Draw bounding box
      ctx.strokeStyle = colorHex;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);

      // Draw semi-transparent fill
      ctx.fillStyle = `rgba(${colorRgb}, 0.1)`;
      ctx.fillRect(x, y, width, height);

      // Draw pulsing dot indicator
      const time = Date.now();
      const pulseOpacity = (Math.sin(time / 200) + 1) / 2; // 0 to 1
      
      ctx.fillStyle = `rgba(${colorRgb}, ${pulseOpacity})`;
      ctx.beginPath();
      ctx.arc(x + 10, y + 10, 5, 0, 2 * Math.PI);
      ctx.fill();
      
      // Label
      ctx.fillStyle = colorHex;
      ctx.font = '14px "JetBrains Mono", monospace';
      ctx.fillText(`${labelPrefix} ${Math.round(prediction.score * 100)}%`, x, y > 20 ? y - 5 : y + 20);
    };

    // Person: Blue
    const personColorHex = '#3B82F6';
    const personColorRgb = '59, 130, 246';
    trackedPeople.forEach(tp => drawBox(tp.detection, personColorHex, personColorRgb, `Person #${tp.trackId}`));

    // Pet: Green
    const petColorHex = '#10B981';
    const petColorRgb = '16, 185, 129';
    trackedPets.forEach(tp => drawBox(tp.detection, petColorHex, petColorRgb, `${tp.detection.class === 'cat' ? 'Cat' : 'Dog'} #${tp.trackId}`));

    // Draw current count on canvas near the 'Live' indicator
    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.fillRect(16, 56, 120, 30);
    ctx.fillStyle = '#3B82F6'; // Blue
    ctx.font = 'bold 14px "JetBrains Mono", monospace';
    ctx.fillText(`人數: ${currentCountRef.current}`, 26, 76);

    if (petEnabled) {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
      ctx.fillRect(16, 94, 120, 30);
      ctx.fillStyle = '#10B981'; // Green
      ctx.font = 'bold 14px "JetBrains Mono", monospace';
      ctx.fillText(`寵物: ${currentPetCountRef.current}`, 26, 114);
    }

    // Draw Heatmap Zones
    if (isEditingZonesRef.current || zonesRef.current.length > 0) {
      zonesRef.current.forEach(zone => {
        ctx.strokeStyle = 'rgba(234, 179, 8, 0.8)'; // Yellow
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(zone.x, zone.y, zone.width, zone.height);
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(234, 179, 8, 0.2)';
        ctx.fillRect(zone.x, zone.y, zone.width, zone.height);
        
        const hits = zoneHitsRef.current[zone.id] || 0;
        const score = Math.min(10, Math.round(hits / 30)); // 1 point per 30 frames
        ctx.fillStyle = 'rgba(234, 179, 8, 0.9)';
        ctx.font = 'bold 16px "JetBrains Mono", monospace';
        ctx.fillText(`${zone.name} (熱度: ${score})`, zone.x, zone.y > 20 ? zone.y - 5 : zone.y + 20);
      });
      
      if (drawingZoneRef.current) {
        const dz = drawingZoneRef.current;
        ctx.strokeStyle = 'rgba(234, 179, 8, 0.8)';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(dz.x!, dz.y!, dz.width!, dz.height!);
        ctx.setLineDash([]);
      }
    }
  };

  const updateCount = (rawPersonCount: number, rawPetCount: number) => {
    const now = Date.now();
    
    // Add to recent counts
    recentCountsRef.current.push({ timestamp: now, count: rawPersonCount });
    recentPetCountsRef.current.push({ timestamp: now, count: rawPetCount });
    
    // Remove counts older than 2 seconds (rolling window)
    recentCountsRef.current = recentCountsRef.current.filter(c => now - c.timestamp <= 2000);
    recentPetCountsRef.current = recentPetCountsRef.current.filter(c => now - c.timestamp <= 2000);
    
    // Calculate highest count in the 2-second window (debouncing)
    const maxPersonCountInWindow = Math.max(...recentCountsRef.current.map(c => c.count), 0);
    const maxPetCountInWindow = Math.max(...recentPetCountsRef.current.map(c => c.count), 0);
    
    setCurrentCount(prev => {
      if (prev !== maxPersonCountInWindow) {
        // Log the change
        const currentZoneScores: Record<string, number> = {};
        zonesRef.current.forEach(z => {
          currentZoneScores[z.name] = Math.min(10, Math.round((zoneHitsRef.current[z.id] || 0) / 30));
        });
        
        setHistory(h => [...h, { 
          timestamp: now, 
          count: maxPersonCountInWindow,
          petCount: maxPetCountInWindow,
          zoneScores: currentZoneScores
        }].slice(-1000));
        
        // Update peak
        setPeakCount(peak => Math.max(peak, maxPersonCountInWindow));
      }
      return maxPersonCountInWindow;
    });

    setCurrentPetCount(maxPetCountInWindow);
  };

  const calculateIoU = (boxA: number[], boxB: number[]) => {
    if (!boxA || !boxB) return 0;
    const xA = Math.max(boxA[0], boxB[0]);
    const yA = Math.max(boxA[1], boxB[1]);
    const xB = Math.min(boxA[0] + boxA[2], boxB[0] + boxB[2]);
    const yB = Math.min(boxA[1] + boxA[3], boxB[1] + boxB[3]);

    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    if (interArea === 0) return 0;

    const boxAArea = boxA[2] * boxA[3];
    const boxBArea = boxB[2] * boxB[3];

    return interArea / (boxAArea + boxBArea - interArea);
  };

  const updateTracking = (people: cocoSsd.DetectedObject[]) => {
    const now = Date.now();
    const MEMORY_DURATION = 10000; // 10 seconds
    const BASE_MAX_DISTANCE = 300;

    const currentTracks = tracksRef.current.filter(t => now - t.lastSeen <= MEMORY_DURATION);
    const detections = people.map(p => ({
      bbox: p.bbox,
      centroid: {
        x: p.bbox[0] + p.bbox[2] / 2,
        y: p.bbox[1] + p.bbox[3] / 2
      },
      original: p
    }));

    const matches: { trackIndex: number, detectionIndex: number, distance: number, iou: number }[] = [];

    detections.forEach((det, dIdx) => {
      currentTracks.forEach((track, tIdx) => {
        const dx = det.centroid.x - track.centroid.x;
        const dy = det.centroid.y - track.centroid.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const iou = calculateIoU(det.bbox, track.bbox);
        
        const timeElapsed = now - track.lastSeen;
        const allowedDistance = BASE_MAX_DISTANCE + (timeElapsed / 1000) * 150;

        if (distance <= allowedDistance || iou > 0.2) {
          // Score: lower is better. We combine distance and IoU.
          const score = distance - (iou * 500);
          matches.push({ trackIndex: tIdx, detectionIndex: dIdx, distance: score, iou });
        }
      });
    });

    matches.sort((a, b) => a.distance - b.distance);

    const matchedTracks = new Set<number>();
    const matchedDetections = new Set<number>();
    const newTracks: Track[] = [];
    const trackedDetections: { detection: cocoSsd.DetectedObject, trackId: number }[] = [];
    let newPeopleCount = 0;

    matches.forEach(match => {
      if (!matchedTracks.has(match.trackIndex) && !matchedDetections.has(match.detectionIndex)) {
        matchedTracks.add(match.trackIndex);
        matchedDetections.add(match.detectionIndex);
        
        const track = currentTracks[match.trackIndex];
        track.centroid = detections[match.detectionIndex].centroid;
        track.bbox = detections[match.detectionIndex].bbox;
        track.lastSeen = now;
        track.hitCount += 1;
        
        if (track.hitCount >= 3 && !track.counted) {
          track.counted = true;
          newPeopleCount++;
          
          const logEntry: DetectionLogEntry = {
            id: `person-${track.id}-${now}`,
            type: 'person',
            timestamp: new Date(now).toLocaleTimeString(),
            bbox: track.bbox,
            score: detections[match.detectionIndex].original.score
          };
          setDetectionLog(prev => [logEntry, ...prev].slice(0, 50));
        }
        newTracks.push(track);
        trackedDetections.push({ detection: detections[match.detectionIndex].original, trackId: track.id });
      }
    });

    detections.forEach((det, dIdx) => {
      if (!matchedDetections.has(dIdx)) {
        const newId = nextTrackIdRef.current++;
        newTracks.push({
          id: newId,
          centroid: det.centroid,
          bbox: det.bbox,
          lastSeen: now,
          hitCount: 1,
          counted: false
        });
        trackedDetections.push({ detection: det.original, trackId: newId });
      }
    });

    currentTracks.forEach((track, tIdx) => {
      if (!matchedTracks.has(tIdx)) {
        newTracks.push(track);
      }
    });

    // Update zone hits for currently detected tracks
    newTracks.forEach(track => {
      if (track.lastSeen === now) {
        zonesRef.current.forEach(zone => {
          if (track.centroid.x >= zone.x && track.centroid.x <= zone.x + zone.width &&
              track.centroid.y >= zone.y && track.centroid.y <= zone.y + zone.height) {
            zoneHitsRef.current[zone.id] = (zoneHitsRef.current[zone.id] || 0) + 1;
          }
        });
      }
    });

    tracksRef.current = newTracks;

    return trackedDetections;
  };

  // Separate tracking for pets
  const petTracksRef = useRef<Track[]>([]);
  const nextPetTrackIdRef = useRef<number>(1);

  const updatePetTracking = (pets: cocoSsd.DetectedObject[]) => {
    if (!isPetCountingEnabledRef.current) return [];

    const now = Date.now();
    const MEMORY_DURATION = 10000;
    const BASE_MAX_DISTANCE = 300;

    const currentTracks = petTracksRef.current.filter(t => now - t.lastSeen <= MEMORY_DURATION);
    const detections = pets.map(p => ({
      bbox: p.bbox,
      centroid: {
        x: p.bbox[0] + p.bbox[2] / 2,
        y: p.bbox[1] + p.bbox[3] / 2
      },
      original: p
    }));

    const matches: { trackIndex: number, detectionIndex: number, distance: number, iou: number }[] = [];

    detections.forEach((det, dIdx) => {
      currentTracks.forEach((track, tIdx) => {
        const dx = det.centroid.x - track.centroid.x;
        const dy = det.centroid.y - track.centroid.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const iou = calculateIoU(det.bbox, track.bbox);
        
        const timeElapsed = now - track.lastSeen;
        const allowedDistance = BASE_MAX_DISTANCE + (timeElapsed / 1000) * 150;

        if (distance <= allowedDistance || iou > 0.2) {
          const score = distance - (iou * 500);
          matches.push({ trackIndex: tIdx, detectionIndex: dIdx, distance: score, iou });
        }
      });
    });

    matches.sort((a, b) => a.distance - b.distance);

    const matchedTracks = new Set<number>();
    const matchedDetections = new Set<number>();
    const newTracks: Track[] = [];
    const trackedDetections: { detection: cocoSsd.DetectedObject, trackId: number }[] = [];
    let newPetCount = 0;

    matches.forEach(match => {
      if (!matchedTracks.has(match.trackIndex) && !matchedDetections.has(match.detectionIndex)) {
        matchedTracks.add(match.trackIndex);
        matchedDetections.add(match.detectionIndex);
        
        const track = currentTracks[match.trackIndex];
        track.centroid = detections[match.detectionIndex].centroid;
        track.bbox = detections[match.detectionIndex].bbox;
        track.lastSeen = now;
        track.hitCount += 1;
        
        if (track.hitCount >= 3 && !track.counted) {
          track.counted = true;
          newPetCount++;
          
          const logEntry: DetectionLogEntry = {
            id: `pet-${track.id}-${now}`,
            type: detections[match.detectionIndex].original.class as 'pet',
            timestamp: new Date(now).toLocaleTimeString(),
            bbox: track.bbox,
            score: detections[match.detectionIndex].original.score
          };
          setDetectionLog(prev => [logEntry, ...prev].slice(0, 50));
        }
        newTracks.push(track);
        trackedDetections.push({ detection: detections[match.detectionIndex].original, trackId: track.id });
      }
    });

    detections.forEach((det, dIdx) => {
      if (!matchedDetections.has(dIdx)) {
        const newId = nextPetTrackIdRef.current++;
        newTracks.push({
          id: newId,
          centroid: det.centroid,
          bbox: det.bbox,
          lastSeen: now,
          hitCount: 1,
          counted: false
        });
        trackedDetections.push({ detection: det.original, trackId: newId });
      }
    });

    currentTracks.forEach((track, tIdx) => {
      if (!matchedTracks.has(tIdx)) {
        newTracks.push(track);
      }
    });

    petTracksRef.current = newTracks;

    if (newPetCount > 0) {
      setTotalPetCountToday(prev => {
        const next = prev + newPetCount;
        localStorage.setItem('totalPetCountToday', String(next));
        return next;
      });
    }

    return trackedDetections;
  };

  // Webhook
  const startWebhookInterval = () => {
    if (webhookIntervalRef.current) {
      window.clearInterval(webhookIntervalRef.current);
    }

    // Every 5 minutes (300,000 ms)
    webhookIntervalRef.current = window.setInterval(() => {
      sendWebhookPayload();
    }, 300000);
  };

  const sendWebhookPayload = async () => {
    if (!isSystemActiveRef.current) {
      setToast({ message: '系統未啟動，無法上傳', type: 'error' });
      return;
    }

    if (!dataApiUrlRef.current && !imageApiUrlRef.current) {
      setToast({ message: '請先設定上傳 API URL', type: 'error' });
      return;
    }

    try {
      const timestamp = new Date().toISOString();
      const promises: Promise<Response>[] = [];
      let dataApiPromise: Promise<Response> | null = null;
      let imageApiPromise: Promise<Response> | null = null;

      // Request 1: Data API
      if (dataApiUrlRef.current) {
        const payload: any = {
          store_code: storeCodeRef.current,
          timestamp: timestamp,
          unique_persons_today: uniquePersonsTodayRef.current,
          peak_people_count: peakCountRef.current,
          total_pets_today: totalPetCountTodayRef.current
        };

        dataApiPromise = fetch(dataApiUrlRef.current, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        promises.push(dataApiPromise);
      }

      // Request 2: Image API
      if (imageApiUrlRef.current && videoRef.current) {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = videoRef.current.videoWidth;
        tempCanvas.height = videoRef.current.videoHeight;
        const ctx = tempCanvas.getContext('2d');
        
        if (ctx) {
          ctx.drawImage(videoRef.current, 0, 0, tempCanvas.width, tempCanvas.height);
          if (heatmapCanvasRef.current) {
            ctx.drawImage(heatmapCanvasRef.current, 0, 0, tempCanvas.width, tempCanvas.height);
          }

          imageApiPromise = new Promise<Response>((resolve, reject) => {
            tempCanvas.toBlob(async (blob) => {
              if (!blob) {
                reject(new Error('Failed to create image blob'));
                return;
              }
              const formData = new FormData();
              formData.append('image', blob, 'heatmap.jpg');
              formData.append('store_code', storeCodeRef.current);
              formData.append('timestamp', timestamp);

              try {
                const res = await fetch(imageApiUrlRef.current!, {
                  method: 'POST',
                  body: formData
                });
                resolve(res);
              } catch (err) {
                reject(err);
              }
            }, 'image/jpeg', 0.8);
          });
          promises.push(imageApiPromise);
        }
      }

      if (promises.length === 0) return;

      const results = await Promise.allSettled(promises);
      
      let dataSuccess = false;
      let imageSuccess = false;

      if (dataApiPromise) {
        const dataResult = results[0];
        if (dataResult.status === 'fulfilled' && dataResult.value.ok) {
          dataSuccess = true;
        }
      } else {
        dataSuccess = true;
      }

      if (imageApiPromise) {
        const imageResult = dataApiPromise ? results[1] : results[0];
        if (imageResult.status === 'fulfilled' && imageResult.value.ok) {
          imageSuccess = true;
        }
      } else {
        imageSuccess = true;
      }

      if (dataSuccess && imageSuccess) {
        setToast({ message: '✅ 上傳成功', type: 'success' });
      } else if (!dataSuccess && !imageSuccess) {
        setToast({ message: '❌ 數據與圖檔上傳皆失敗', type: 'error' });
      } else if (!dataSuccess) {
        setToast({ message: '❌ 數據上傳失敗', type: 'error' });
      } else {
        setToast({ message: '❌ 圖檔上傳失敗', type: 'error' });
      }

    } catch (error) {
      console.error("Upload error:", error);
      setToast({ message: '❌ 上傳發生未預期錯誤', type: 'error' });
    }
  };

  const handleSaveWebhook = () => {
    localStorage.setItem('dataApiUrl', dataApiUrl);
    localStorage.setItem('imageApiUrl', imageApiUrl);
    setToast({ message: 'API 設定已儲存', type: 'success' });
  };

  const downloadHeatmap = () => {
    if (!videoRef.current || !heatmapCanvasRef.current) return;
    
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = videoRef.current.videoWidth;
    tempCanvas.height = videoRef.current.videoHeight;
    const ctx = tempCanvas.getContext('2d');
    
    if (ctx) {
      ctx.drawImage(videoRef.current, 0, 0, tempCanvas.width, tempCanvas.height);
      ctx.drawImage(heatmapCanvasRef.current, 0, 0, tempCanvas.width, tempCanvas.height);
      
      const link = document.createElement('a');
      link.download = `heatmap-${new Date().toISOString().replace(/[:.]/g, '-')}.jpg`;
      link.href = tempCanvas.toDataURL('image/jpeg', 0.8);
      link.click();
    }
  };

  const clearHeatmap = () => {
    if (heatmapCanvasRef.current) {
      const ctx = heatmapCanvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, heatmapCanvasRef.current.width, heatmapCanvasRef.current.height);
      }
    }
    setToast({ message: '熱點資料已清除', type: 'success' });
  };

  // Export CSV
  const exportCSV = () => {
    const now = new Date();
    const zoneNames = zones.map(z => z.name);
    const headers = ['時間', '店櫃代號', '人數', '寵物數', ...zoneNames.map(n => `熱點區域_${n}_分數(0-10)`)];
    
    const rows = history.map(stat => {
      const date = new Date(stat.timestamp);
      const formattedDate = `${date.getFullYear()}/${String(date.getMonth() + 1).padStart(2, '0')}/${String(date.getDate()).padStart(2, '0')} ${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}:${String(date.getSeconds()).padStart(2, '0')}`;
      
      const row = [
        formattedDate,
        storeCode || '未設定',
        stat.count.toString(),
        (stat.petCount || 0).toString()
      ];
      
      zoneNames.forEach(name => {
        row.push((stat.zoneScores?.[name] || 0).toString());
      });
      
      return row;
    });

    const csvContent = [
      headers.join(','),
      ...rows.map(e => e.join(','))
    ].join('\n');

    const blob = new Blob(["\uFEFF" + csvContent], { type: 'text/csv;charset=utf-8;' }); // \uFEFF for Excel UTF-8 BOM
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    
    const safeStoreCode = storeCode.trim() || '未設定';
    const fileDate = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}`;
    link.setAttribute('download', `${safeStoreCode}_${fileDate}_summary.csv`);
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleZoneMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isEditingZones || !videoRef.current) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const scaleX = videoRef.current.videoWidth / rect.width;
    const scaleY = videoRef.current.videoHeight / rect.height;
    const x = e.nativeEvent.offsetX * scaleX;
    const y = e.nativeEvent.offsetY * scaleY;
    setDrawingZone({ x, y, width: 0, height: 0 });
  };

  const handleZoneMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isEditingZones || !drawingZone || !videoRef.current) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const scaleX = videoRef.current.videoWidth / rect.width;
    const scaleY = videoRef.current.videoHeight / rect.height;
    const currentX = e.nativeEvent.offsetX * scaleX;
    const currentY = e.nativeEvent.offsetY * scaleY;
    setDrawingZone(prev => ({
      ...prev,
      width: currentX - prev!.x!,
      height: currentY - prev!.y!
    }));
  };

  const handleZoneMouseUp = () => {
    if (!isEditingZones || !drawingZone) return;
    if (Math.abs(drawingZone.width!) > 20 && Math.abs(drawingZone.height!) > 20) {
      const normalizedZone = {
        id: Date.now().toString(),
        name: `區域 ${zones.length + 1}`,
        x: drawingZone.width! < 0 ? drawingZone.x! + drawingZone.width! : drawingZone.x!,
        y: drawingZone.height! < 0 ? drawingZone.y! + drawingZone.height! : drawingZone.y!,
        width: Math.abs(drawingZone.width!),
        height: Math.abs(drawingZone.height!)
      };
      setZones(prev => [...prev, normalizedZone as HeatmapZone]);
    }
    setDrawingZone(null);
  };

  const removeZone = (id: string) => {
    setZones(prev => prev.filter(z => z.id !== id));
    const newHits = { ...zoneHitsRef.current };
    delete newHits[id];
    zoneHitsRef.current = newHits;
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopSystem();
    };
  }, []);

  // Prepare chart data
  const chartData = history.slice(-50).map(h => ({
    time: new Date(h.timestamp).toLocaleTimeString('zh-TW', { hour12: false }),
    人數: h.count
  }));

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-gray-100 font-sans selection:bg-emerald-500/30">
      {/* Header */}
      <header className="border-b border-white/10 bg-[#141414] px-6 py-4 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold tracking-tight text-white flex items-center gap-2">
              <Activity className="w-5 h-5 text-emerald-500" />
              智能人流分析系統
            </h1>
            <p className="text-sm text-gray-400 mt-1">即時影像辨識與數據監控</p>
          </div>
          
          <div className="flex items-center gap-4">
            {availableCameras.length > 0 && (
              <select
                value={selectedCameraId}
                onChange={handleCameraChange}
                className="bg-[#141414] border border-white/10 text-white text-sm rounded-lg focus:ring-emerald-500 focus:border-emerald-500 block p-2.5 max-w-[200px] truncate"
              >
                {availableCameras.map((camera, index) => (
                  <option key={camera.deviceId || index} value={camera.deviceId}>
                    {camera.label || `Camera ${index + 1}`}
                  </option>
                ))}
              </select>
            )}
            <button
              onClick={toggleSystem}
              disabled={isModelLoading}
              className={cn(
                "relative inline-flex h-10 items-center justify-center gap-2 rounded-lg px-6 font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-[#0a0a0a]",
                isModelLoading ? "bg-gray-800 text-gray-500 cursor-not-allowed" :
                isSystemActive 
                  ? "bg-red-500/10 text-red-500 hover:bg-red-500/20 focus:ring-red-500" 
                  : "bg-emerald-500 text-white hover:bg-emerald-600 focus:ring-emerald-500"
              )}
            >
              {isModelLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-gray-500 border-t-transparent rounded-full animate-spin" />
                  模型載入中...
                </>
              ) : isSystemActive ? (
                <>
                  <CameraOff className="w-4 h-4" />
                  停止系統
                </>
              ) : (
                <>
                  <Camera className="w-4 h-4" />
                  啟動系統
                </>
              )}
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Column: Video & Settings */}
        <div className="lg:col-span-2 space-y-8">
          
          {/* Video Container */}
          <div 
            className="relative bg-[#141414] rounded-2xl border border-white/5 overflow-hidden shadow-2xl aspect-video flex items-center justify-center"
          >
            {isEditingZones && (
              <div 
                className="absolute inset-0 z-30 cursor-crosshair"
                onMouseDown={handleZoneMouseDown}
                onMouseMove={handleZoneMouseMove}
                onMouseUp={handleZoneMouseUp}
                onMouseLeave={handleZoneMouseUp}
              />
            )}
            {!isSystemActive && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-500 z-10 bg-[#141414]">
                <CameraOff className="w-12 h-12 mb-4 opacity-50" />
                <p className="text-lg font-medium">攝影機已關閉</p>
                <p className="text-sm mt-2 opacity-70">點擊右上角「啟動系統」開始偵測</p>
              </div>
            )}
            
            <video
              ref={videoRef}
              className="absolute inset-0 w-full h-full object-cover"
              playsInline
              muted
            />
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full object-cover z-20 pointer-events-none"
            />
            <canvas
              ref={heatmapCanvasRef}
              className="absolute inset-0 w-full h-full object-cover z-20 pointer-events-none"
            />
            
            {/* Live Indicator */}
            {isSystemActive && (
              <div className="absolute top-4 left-4 z-30 flex items-center gap-2 bg-black/50 backdrop-blur-md px-3 py-1.5 rounded-full border border-white/10">
                <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                <span className="text-xs font-mono font-medium tracking-wider text-white uppercase">Live</span>
              </div>
            )}
          </div>

          {/* Settings & Export */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* System Settings */}
            <div className="bg-[#141414] rounded-xl border border-white/5 p-6 md:col-span-2">
              <div 
                className="flex items-center justify-between cursor-pointer select-none"
                onClick={() => setIsSettingsOpen(!isSettingsOpen)}
              >
                <div className="flex items-center gap-2">
                  <Settings className="w-5 h-5 text-gray-400" />
                  <h2 className="text-lg font-medium text-white">系統設定</h2>
                </div>
                {isSettingsOpen ? (
                  <ChevronUp className="w-5 h-5 text-gray-400" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-gray-400" />
                )}
              </div>
              
              {isSettingsOpen && (
                <div className="space-y-6 mt-6 animate-in fade-in slide-in-from-top-2">
                  {/* Camera Selection */}
                  <div className="space-y-3">
                    <div>
                      <h3 className="text-white font-medium">選擇攝影鏡頭</h3>
                      <p className="text-sm text-gray-400 mt-1">請選擇要用於影像辨識的攝影機。</p>
                    </div>
                    <select
                      value={selectedCameraId}
                      onChange={(e) => switchCamera(e.target.value)}
                      className="w-full bg-[#0a0a0a] border border-white/10 rounded-lg px-4 py-2.5 text-sm text-white focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition-colors appearance-none"
                    >
                      {availableCameras.map((camera) => (
                        <option key={camera.deviceId} value={camera.deviceId}>
                          {camera.label || `Camera ${camera.deviceId.substring(0, 5)}...`}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Store Code */}
                  <div className="space-y-3">
                  <div>
                    <h3 className="text-white font-medium">店櫃代號 <span className="text-red-500">*</span></h3>
                    <p className="text-sm text-gray-400 mt-1">必須輸入店櫃代號才可以啟動系統。</p>
                  </div>
                  <input
                    type="text"
                    value={storeCode}
                    onChange={(e) => {
                      setStoreCode(e.target.value);
                      localStorage.setItem('storeCode', e.target.value);
                    }}
                    placeholder="請輸入店櫃代號"
                    className="w-full bg-[#0a0a0a] border border-white/10 rounded-lg px-4 py-2.5 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition-colors"
                  />
                </div>

                {/* Pet Counting */}
                <div className="flex items-center justify-between p-4 bg-[#0a0a0a] rounded-lg border border-white/5">
                  <div>
                    <h3 className="text-white font-medium">獨立計算寵物</h3>
                    <p className="text-sm text-gray-400 mt-1">開啟後將分別辨識並計算畫面中的貓與狗。</p>
                  </div>
                  <button
                    onClick={() => {
                      setIsPetCountingEnabled(prev => {
                        const next = !prev;
                        localStorage.setItem('isPetCountingEnabled', String(next));
                        return next;
                      });
                    }}
                    className={cn(
                      "relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 focus:ring-offset-[#0a0a0a]",
                      isPetCountingEnabled ? "bg-emerald-500" : "bg-gray-600"
                    )}
                  >
                    <span
                      className={cn(
                        "inline-block h-4 w-4 transform rounded-full bg-white transition-transform",
                        isPetCountingEnabled ? "translate-x-6" : "translate-x-1"
                      )}
                    />
                  </button>
                </div>

                {/* Dynamic AI Detection Interval */}
                <div className="flex flex-col space-y-3 p-4 bg-[#0a0a0a] rounded-lg border border-white/5">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-white font-medium">AI 偵測頻率 (毫秒)</h3>
                      <p className="text-sm text-gray-400 mt-1">調整 AI 辨識的執行間隔，數值越低越即時，但會增加運算負擔。</p>
                    </div>
                    <span className="text-emerald-500 font-mono font-medium">{detectionIntervalMs} ms</span>
                  </div>
                  <input
                    type="range"
                    min="300"
                    max="5000"
                    step="100"
                    value={detectionIntervalMs}
                    onChange={(e) => {
                      const val = parseInt(e.target.value, 10);
                      setDetectionIntervalMs(val);
                      localStorage.setItem('detectionIntervalMs', String(val));
                    }}
                    className="w-full accent-emerald-500"
                  />
                  <div className="flex justify-between text-xs text-gray-500 font-mono">
                    <span>300ms</span>
                    <span>5000ms</span>
                  </div>
                </div>

                {/* Heatmap Analytics */}
                <div className="space-y-4 pt-4 border-t border-white/5">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-white font-medium">熱點分析</h3>
                      <p className="text-sm text-gray-400 mt-1">記錄顧客在畫面中的停留軌跡。</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between p-4 bg-[#0a0a0a] rounded-lg border border-white/5">
                    <div>
                      <h3 className="text-white font-medium text-sm">啟用/暫停熱點記錄</h3>
                    </div>
                    <button
                      onClick={() => setIsRecordingHeatmap(!isRecordingHeatmap)}
                      className={cn(
                        "relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 focus:ring-offset-[#0a0a0a]",
                        isRecordingHeatmap ? "bg-emerald-500" : "bg-gray-600"
                      )}
                    >
                      <span
                        className={cn(
                          "inline-block h-4 w-4 transform rounded-full bg-white transition-transform",
                          isRecordingHeatmap ? "translate-x-6" : "translate-x-1"
                        )}
                      />
                    </button>
                  </div>
                  
                  <button
                    onClick={clearHeatmap}
                    className="w-full flex items-center justify-center gap-2 bg-white/5 hover:bg-white/10 text-white rounded-lg px-4 py-2.5 text-sm font-medium transition-colors border border-white/5"
                  >
                    清除熱點資料
                  </button>

                  <div className="pt-4 border-t border-white/5">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-sm text-gray-300">自訂熱點區域</span>
                      <button
                        onClick={() => setIsEditingZones(!isEditingZones)}
                        className={cn("px-3 py-1.5 text-sm rounded-lg transition-colors", isEditingZones ? "bg-yellow-500/20 text-yellow-500 border border-yellow-500/50" : "bg-white/5 text-gray-300 hover:bg-white/10")}
                      >
                        {isEditingZones ? '完成設定' : '設定區域'}
                      </button>
                    </div>
                    {isEditingZones && (
                      <p className="text-xs text-yellow-500 mb-3">請在左側影像畫面上拖曳滑鼠來建立區域。</p>
                    )}
                    {zones.length > 0 && (
                      <div className="space-y-2 max-h-40 overflow-y-auto pr-1 custom-scrollbar">
                        {zones.map(zone => {
                          const score = Math.min(10, Math.round((zoneHitsRef.current[zone.id] || 0) / 30));
                          return (
                            <div key={zone.id} className="flex items-center justify-between bg-black/50 p-2 rounded border border-white/5">
                              <span className="text-sm text-gray-300">{zone.name}</span>
                              <div className="flex items-center gap-3">
                                <span className="text-sm text-yellow-500 font-mono">分數: {score}</span>
                                <button onClick={() => removeZone(zone.id)} className="text-red-400 hover:text-red-300 transition-colors">
                                  <Trash2 className="w-4 h-4" />
                                </button>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                </div>

                {/* Advanced Settings */}
                <div className="space-y-4 pt-4 border-t border-white/5">
                  <button 
                    onClick={() => setIsAdvancedSettingsOpen(!isAdvancedSettingsOpen)}
                    className="flex items-center justify-between w-full text-left"
                  >
                    <div>
                      <h3 className="text-white font-medium">進階設定</h3>
                      <p className="text-sm text-gray-400 mt-1">調整辨識信心度與隱私設定。</p>
                    </div>
                    {isAdvancedSettingsOpen ? (
                      <ChevronUp className="w-5 h-5 text-gray-400" />
                    ) : (
                      <ChevronDown className="w-5 h-5 text-gray-400" />
                    )}
                  </button>
                  
                  {isAdvancedSettingsOpen && (
                    <div className="space-y-3 pt-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-300">人臉模糊 (隱私保護)</span>
                        <button
                          onClick={() => {
                            setIsFaceBlurEnabled(!isFaceBlurEnabled);
                            localStorage.setItem('isFaceBlurEnabled', String(!isFaceBlurEnabled));
                          }}
                          className={cn(
                            "relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 focus:ring-offset-[#0a0a0a]",
                            isFaceBlurEnabled ? "bg-emerald-500" : "bg-gray-600"
                          )}
                        >
                          <span className={cn("inline-block h-4 w-4 transform rounded-full bg-white transition-transform", isFaceBlurEnabled ? "translate-x-6" : "translate-x-1")} />
                        </button>
                      </div>

                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-300">人員辨識信心度</span>
                          <span className="text-sm text-emerald-500 font-mono">{Math.round(personConfidence * 100)}%</span>
                        </div>
                        <input
                          type="range"
                          min="0.3"
                          max="0.9"
                          step="0.05"
                          value={personConfidence}
                          onChange={(e) => {
                            const val = parseFloat(e.target.value);
                            setPersonConfidence(val);
                            localStorage.setItem('personConfidence', String(val));
                          }}
                          className="w-full accent-emerald-500"
                        />
                      </div>

                      {isPetCountingEnabled && (
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <span className="text-sm text-gray-300">寵物辨識信心度</span>
                            <span className="text-sm text-emerald-500 font-mono">{Math.round(petConfidence * 100)}%</span>
                          </div>
                          <input
                            type="range"
                            min="0.3"
                            max="0.9"
                            step="0.05"
                            value={petConfidence}
                            onChange={(e) => {
                              const val = parseFloat(e.target.value);
                              setPetConfidence(val);
                              localStorage.setItem('petConfidence', String(val));
                            }}
                            className="w-full accent-emerald-500"
                          />
                        </div>
                      )}

                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-300">人臉比對閾值 (越小越嚴格)</span>
                          <span className="text-sm text-emerald-500 font-mono">{faceMatchThreshold.toFixed(2)}</span>
                        </div>
                        <input
                          type="range"
                          min="0.3"
                          max="0.8"
                          step="0.05"
                          value={faceMatchThreshold}
                          onChange={(e) => {
                            const val = parseFloat(e.target.value);
                            setFaceMatchThreshold(val);
                            localStorage.setItem('faceMatchThreshold', String(val));
                          }}
                          className="w-full accent-emerald-500"
                        />
                      </div>
                    </div>
                  )}
                </div>

                {/* Webhook */}
                <div className="space-y-3 pt-4 border-t border-white/5">
                  <div>
                    <h3 className="text-white font-medium">數據上傳 API (Data URL)</h3>
                    <p className="text-sm text-gray-400 mt-1">傳送 JSON 格式的統計數據。</p>
                  </div>
                  <input
                    type="url"
                    value={dataApiUrl}
                    onChange={(e) => setDataApiUrl(e.target.value)}
                    placeholder="https://your-data-api.com/endpoint"
                    className="w-full bg-[#0a0a0a] border border-white/10 rounded-lg px-4 py-2.5 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition-colors"
                  />
                  
                  <div className="pt-2">
                    <h3 className="text-white font-medium">圖檔上傳 API (Image URL)</h3>
                    <p className="text-sm text-gray-400 mt-1">傳送包含熱點圖的截圖 (FormData)。</p>
                  </div>
                  <input
                    type="url"
                    value={imageApiUrl}
                    onChange={(e) => setImageApiUrl(e.target.value)}
                    placeholder="https://your-image-api.com/endpoint"
                    className="w-full bg-[#0a0a0a] border border-white/10 rounded-lg px-4 py-2.5 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition-colors"
                  />
                  
                  <button
                    onClick={handleSaveWebhook}
                    className="w-full flex items-center justify-center gap-2 bg-white/5 hover:bg-white/10 text-white rounded-lg px-4 py-2.5 text-sm font-medium transition-colors border border-white/5"
                  >
                    <Save className="w-4 h-4" />
                    儲存設定
                  </button>
                  
                  <div className="pt-4 border-t border-white/5">
                    <h3 className="text-white font-medium mb-3">熱點截圖上傳</h3>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        onClick={sendWebhookPayload}
                        className="flex items-center justify-center gap-2 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-500 rounded-lg px-4 py-2.5 text-sm font-medium transition-colors border border-emerald-500/20"
                      >
                        <Camera className="w-4 h-4" />
                        上傳圖檔
                      </button>
                      <button
                        onClick={downloadHeatmap}
                        className="flex items-center justify-center gap-2 bg-blue-500/10 hover:bg-blue-500/20 text-blue-500 rounded-lg px-4 py-2.5 text-sm font-medium transition-colors border border-blue-500/20"
                      >
                        <Download className="w-4 h-4" />
                        直接下載
                      </button>
                    </div>
                  </div>
                </div>
              </div>
              )}
            </div>

            {/* Data Export & Chart */}
            <div className="bg-[#141414] rounded-xl border border-white/5 p-6 md:col-span-2">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
                <div className="flex items-center gap-2">
                  <Download className="w-5 h-5 text-gray-400" />
                  <h2 className="text-lg font-medium text-white">數據匯出與歷史趨勢</h2>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => setIsLogModalOpen(true)}
                    className="flex items-center justify-center gap-2 bg-white/5 hover:bg-white/10 text-white rounded-lg px-4 py-2 text-sm font-medium transition-colors border border-white/5"
                  >
                    <List className="w-4 h-4" />
                    檢視辨識紀錄
                  </button>
                  <button
                    onClick={exportCSV}
                    className="flex items-center justify-center gap-2 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-500 rounded-lg px-4 py-2 text-sm font-medium transition-colors border border-emerald-500/20"
                  >
                    <Download className="w-4 h-4" />
                    匯出 CSV 檔
                  </button>
                </div>
              </div>
              <p className="text-sm text-gray-400 mb-6">檢視最近的偵測數據趨勢，或下載完整的歷史紀錄（最多保留 1000 筆）。</p>
              
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    <XAxis dataKey="time" stroke="#888" fontSize={12} />
                    <YAxis stroke="#888" fontSize={12} allowDecimals={false} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#141414', borderColor: '#333', color: '#fff' }}
                      itemStyle={{ color: '#10B981' }}
                    />
                    <Line type="monotone" dataKey="人數" stroke="#10B981" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Metrics */}
        <div className="space-y-6">
          {/* Current People Card */}
          <div className="bg-[#141414] rounded-2xl border border-white/5 p-6 relative overflow-hidden group">
            <div className={cn(
              "absolute inset-0 bg-gradient-to-br to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500",
              isPetCountingEnabled ? "from-blue-500/5" : "from-emerald-500/5"
            )} />
            <div className="flex items-center justify-between mb-4 relative z-10">
              <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider">當前畫面人數</h3>
              <div className={cn(
                "p-2 rounded-lg",
                isPetCountingEnabled ? "bg-blue-500/10" : "bg-emerald-500/10"
              )}>
                <Users className={cn("w-5 h-5", isPetCountingEnabled ? "text-blue-500" : "text-emerald-500")} />
              </div>
            </div>
            <div className="flex items-baseline gap-2 relative z-10">
              <span className="text-6xl font-light tracking-tighter text-white font-mono">
                {currentCount}
              </span>
              <span className="text-gray-500 font-medium">人</span>
            </div>
          </div>

          {/* Current Pet Card (Conditional) */}
          {isPetCountingEnabled && (
            <div className="bg-[#141414] rounded-2xl border border-white/5 p-6 relative overflow-hidden group animate-in fade-in slide-in-from-top-4">
              <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              <div className="flex items-center justify-between mb-4 relative z-10">
                <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider">當前畫面寵物數</h3>
                <div className="p-2 bg-emerald-500/10 rounded-lg">
                  <Dog className="w-5 h-5 text-emerald-500" />
                </div>
              </div>
              <div className="flex items-baseline gap-2 relative z-10">
                <span className="text-6xl font-light tracking-tighter text-white font-mono">
                  {currentPetCount}
                </span>
                <span className="text-gray-500 font-medium">隻</span>
              </div>
            </div>
          )}

          {/* Total Count Today Card */}
          <div className="bg-[#141414] rounded-2xl border border-white/5 p-6 relative overflow-hidden group">
            <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <div className="flex items-center justify-between mb-4 relative z-10">
              <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider">今日不重複人數</h3>
              <div className="p-2 bg-purple-500/10 rounded-lg">
                <UserPlus className="w-5 h-5 text-purple-500" />
              </div>
            </div>
            <div className="flex items-baseline gap-2 relative z-10">
              <span className="text-5xl font-light tracking-tighter text-white font-mono">
                {uniquePersonsToday}
              </span>
              <span className="text-gray-500 font-medium">人</span>
            </div>
          </div>

          {/* Total Pet Count Today Card (Conditional) */}
          {isPetCountingEnabled && (
            <div className="bg-[#141414] rounded-2xl border border-white/5 p-6 relative overflow-hidden group animate-in fade-in slide-in-from-top-4">
              <div className="absolute inset-0 bg-gradient-to-br from-yellow-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              <div className="flex items-center justify-between mb-4 relative z-10">
                <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider">累計寵物數</h3>
                <div className="p-2 bg-yellow-500/10 rounded-lg">
                  <Dog className="w-5 h-5 text-yellow-500" />
                </div>
              </div>
              <div className="flex items-baseline gap-2 relative z-10">
                <span className="text-5xl font-light tracking-tighter text-white font-mono">
                  {totalPetCountToday}
                </span>
                <span className="text-gray-500 font-medium">隻</span>
              </div>
            </div>
          )}

          {/* Peak Count Card */}
          <div className="bg-[#141414] rounded-2xl border border-white/5 p-6 relative overflow-hidden group">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <div className="flex items-center justify-between mb-4 relative z-10">
              <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider">今日最高人數</h3>
              <div className="p-2 bg-blue-500/10 rounded-lg">
                <TrendingUp className="w-5 h-5 text-blue-500" />
              </div>
            </div>
            <div className="flex items-baseline gap-2 relative z-10">
              <span className="text-5xl font-light tracking-tighter text-white font-mono">
                {peakCount}
              </span>
              <span className="text-gray-500 font-medium">人</span>
            </div>
          </div>

          {/* Status Card */}
          <div className="bg-[#141414] rounded-2xl border border-white/5 p-6">
            <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-4">系統狀態</h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">攝影機</span>
                <div className="flex items-center gap-2">
                  <div className={cn("w-2 h-2 rounded-full", isSystemActive ? "bg-emerald-500" : "bg-gray-600")} />
                  <span className={cn("text-sm font-medium", isSystemActive ? "text-emerald-500" : "text-gray-500")}>
                    {isSystemActive ? '運作中' : '已停止'}
                  </span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">AI 模型</span>
                <div className="flex items-center gap-2">
                  <div className={cn("w-2 h-2 rounded-full", !isModelLoading ? "bg-emerald-500" : "bg-amber-500")} />
                  <span className={cn("text-sm font-medium", !isModelLoading ? "text-emerald-500" : "text-amber-500")}>
                    {!isModelLoading ? '已載入 (COCO-SSD)' : '載入中...'}
                  </span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">上傳 API</span>
                <div className="flex items-center gap-2">
                  <div className={cn("w-2 h-2 rounded-full", (dataApiUrl || imageApiUrl) ? "bg-blue-500" : "bg-gray-600")} />
                  <span className={cn("text-sm font-medium", (dataApiUrl || imageApiUrl) ? "text-blue-500" : "text-gray-500")}>
                    {(dataApiUrl || imageApiUrl) ? '已設定' : '未設定'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Toast Notifications */}
      {toast && (
        <Toast 
          message={toast.message} 
          type={toast.type} 
          onClose={() => setToast(null)} 
        />
      )}

      {/* Detection Log Modal */}
      {isLogModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
          <div className="bg-[#141414] border border-white/10 rounded-2xl w-full max-w-3xl max-h-[80vh] flex flex-col shadow-2xl">
            <div className="flex items-center justify-between p-6 border-b border-white/5">
              <div className="flex items-center gap-3">
                <List className="w-5 h-5 text-emerald-500" />
                <h2 className="text-xl font-medium text-white">辨識歷史紀錄</h2>
              </div>
              <button 
                onClick={() => setIsLogModalOpen(false)}
                className="p-2 text-gray-400 hover:text-white rounded-lg hover:bg-white/5 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-6">
              {detectionLog.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-40 text-gray-500">
                  <Activity className="w-8 h-8 mb-3 opacity-50" />
                  <p>尚無辨識紀錄</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {detectionLog.map((log) => (
                    <div key={log.id} className="flex items-center justify-between p-4 bg-[#0a0a0a] border border-white/5 rounded-xl">
                      <div className="flex items-center gap-4">
                        <div className={cn(
                          "w-10 h-10 rounded-full flex items-center justify-center",
                          log.type === 'person' ? "bg-blue-500/10 text-blue-500" : "bg-emerald-500/10 text-emerald-500"
                        )}>
                          {log.type === 'person' ? <Users className="w-5 h-5" /> : <Dog className="w-5 h-5" />}
                        </div>
                        <div>
                          <p className="text-white font-medium capitalize">{log.type === 'person' ? '人員' : (log.type === 'cat' ? '貓' : '狗')}</p>
                          <p className="text-sm text-gray-500 font-mono">{log.timestamp}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-sm text-emerald-500 font-mono">{Math.round(log.score * 100)}% 信心度</p>
                        <p className="text-xs text-gray-600 font-mono mt-1">
                          [{Math.round(log.bbox[0])}, {Math.round(log.bbox[1])}, {Math.round(log.bbox[2])}, {Math.round(log.bbox[3])}]
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
            
            <div className="p-6 border-t border-white/5 bg-[#0a0a0a] rounded-b-2xl flex justify-between items-center">
              <p className="text-sm text-gray-500">顯示最近 50 筆紀錄</p>
              <button
                onClick={() => setDetectionLog([])}
                className="px-4 py-2 text-sm text-red-400 hover:text-red-300 hover:bg-red-400/10 rounded-lg transition-colors"
              >
                清除紀錄
              </button>
            </div>
          </div>
        </div>
      )}
            {/* Footer */}
      <footer className="p-6 border-t border-white/5 bg-[#0a0a0a] rounded-b-2xl flex justify-between items-center">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-[#00000]">智能人流分析系統</span>
          </div>
          <p className="text-sm text-[#00000]">© 2026 奇哥股份有限公司。保留所有權利。</p>
          <div className="flex gap-6 text-sm font-medium text-[#00000]">
            <a href="#" className="hover:text-[#F472B6]">隱私政策</a>
            <a href="#" className="hover:text-[#F472B6]">服務條款</a>
            <a href="#" className="hover:text-[#F472B6]">聯絡我們</a>
          </div>
      </footer>
    </div>
  );
}
