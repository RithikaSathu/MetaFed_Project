// API Client for MetaFed Backend
const API_BASE = 'http://localhost:5000';

const fetchWithTimeout = async (url: string, options: RequestInit = {}, timeout = 10000) => {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(id);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  } catch (err) {
    clearTimeout(id);
    throw err;
  }
};

export const api = {
  // Health check
  health: async () => {
    return fetchWithTimeout(`${API_BASE}/api/health`);
  },

  // Preprocessing
  preprocess: async () => {
    return fetchWithTimeout(`${API_BASE}/api/preprocess`, { method: 'POST' }, 60000);
  },

  // Run homogeneous experiments (FedAvg, FedBN, FedProx, MetaFed with CNN)
  runHomogeneous: async () => {
    return fetchWithTimeout(`${API_BASE}/api/run/homogeneous`, { method: 'POST' }, 300000);
  },

  // Run heterogeneous MetaFed (CNN, RNN, ViT)
  runHeterogeneous: async () => {
    return fetchWithTimeout(`${API_BASE}/api/run/heterogeneous`, { method: 'POST' }, 300000);
  },

  // Get training status
  getStatus: async () => {
    return fetchWithTimeout(`${API_BASE}/api/train/status`);
  },

  // Get all results
  getResults: async () => {
    return fetchWithTimeout(`${API_BASE}/api/results`);
  },

  // Image upload for prediction
  uploadImage: async (file: File, algorithm: string) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('algorithm', algorithm);
    return fetchWithTimeout(`${API_BASE}/api/image-upload`, {
      method: 'POST',
      body: formData,
    }, 30000);
  },

  // Get plot URL
  getPlotUrl: (filename: string) => `${API_BASE}/static/plots/${filename}`,
};

// Constants
export const ALGORITHMS = {
  FEDAVG: 'fedavg',
  FEDPROX: 'fedprox',
  FEDBN: 'fedbn',
  METAFED_HOM: 'metafed_hom',
  METAFED_HET: 'metafed_het',
} as const;

export const ALGORITHM_NAMES: Record<string, string> = {
  [ALGORITHMS.FEDAVG]: 'FedAvg',
  [ALGORITHMS.FEDPROX]: 'FedProx',
  [ALGORITHMS.FEDBN]: 'FedBN',
  [ALGORITHMS.METAFED_HOM]: 'MetaFed (Homogeneous)',
  [ALGORITHMS.METAFED_HET]: 'MetaFed (Heterogeneous)',
};

export const METRICS = ['accuracy', 'precision', 'recall', 'f1'] as const;

export const FEDERATIONS = ['fed_0', 'fed_1', 'fed_2'] as const;

export const MODELS = ['CNN', 'RNN', 'ViT'] as const;

export const DATASET_INFO = {
  name: 'PAMAP2',
  fullName: 'Physical Activity Monitoring Dataset',
  description: 'Physical Activity Monitoring using Accelerometers and Gyroscopes from body-worn sensors',
  subjects: 9,
  activities: 12,
  imuChannels: 27,
  samplingRate: '100 Hz',
  windowSize: 100,
  federations: 3,
  activityLabels: [
    'Lying', 'Sitting', 'Standing', 'Walking', 'Running', 'Cycling',
    'Nordic Walking', 'Ascending Stairs', 'Descending Stairs',
    'Vacuum Cleaning', 'Ironing', 'Rope Jumping'
  ],
};
