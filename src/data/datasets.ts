export interface DatasetEntry {
  id: string;
  taskNumber: number;
  taskName: string;
  datasetName: string;
  role: 'Pretrained Backbone Dataset' | 'Bundled Taxonomy / Manifest' | 'Evaluated External Candidate';
  modelUsed: string;
  modelArtifact: string;
  weightsSha256: string | null;
  labelsCovered: string[];
  license: string;
  sourceUrl: string;
  modelUrl: string;
  notes: string;
  limitations: string;
}

export const WILDLIFE_TAXONOMY_JSON = {
  version: '1.0.0',
  source: 'Species labels and class coverage: COCO 2017 object categories; diet categories are documented below.',
  animal_classes: [
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
  ],
  classifications: {
    bird: 'unknown',
    cat: 'carnivore',
    dog: 'unknown',
    horse: 'non_carnivore',
    sheep: 'non_carnivore',
    cow: 'non_carnivore',
    elephant: 'non_carnivore',
    bear: 'unknown',
    zebra: 'non_carnivore',
    giraffe: 'non_carnivore',
  } as Record<string, 'carnivore' | 'non_carnivore' | 'unknown'>,
  notes: {
    bird: 'Broad COCO class spanning many species and diets; left unclassified.',
    dog: 'Domestic dogs are dietary generalists; the broad label does not justify a strict diet assignment.',
    bear: 'The COCO label does not identify a bear species; diets vary substantially among species.',
  },
  references: [
    {
      title: 'COCO dataset and object categories',
      url: 'https://cocodataset.org/#home',
    },
    {
      title: 'Carnivore',
      url: 'https://www.britannica.com/animal/carnivore',
    },
    {
      title: 'Cats',
      url: 'https://www.britannica.com/animal/cat',
    },
  ],
};

export const SIGN_LANGUAGE_VOCABULARY_JSON = {
  schema_version: '1.0.0',
  sign_language: 'American Sign Language (ASL)',
  recognition_scope: 'static fingerspelled alphabet letters',
  labels: [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y',
  ],
  excluded_dynamic_letters: ['J', 'Z'],
  translation_scope: 'No words, sentences, grammar, or continuous translation',
};

export const SIGN_LANGUAGE_MODEL_MANIFEST_JSON = {
  schema_version: '1.0.0',
  model_id: 'AmimulBmeIU/asl-alphabet-classifier',
  revision: '9e5cde38db66ae889322aeea3c547848676fa96e',
  filename: 'model.keras',
  size_bytes: 245970,
  sha256: '9A73CD144A41C2CA92BD4D07ECEC2FE32F5F4DDE5E28C8BCC0B0ADA013502F11',
  declared_license: 'MIT (publisher-declared; review underlying training data terms separately)',
  architecture: 'Keras classifier over flattened x,y,z coordinates for 21 MediaPipe hand landmarks',
  model_input: [21, 3],
  vocabulary_file: 'vocabulary.json',
  training_dataset: null,
  training_data_provenance: 'Not identified in the publisher model card',
  reported_validation_accuracy: 0.9415,
  reported_metric_status: 'Publisher claim; dataset, signer split, and evaluation protocol are not documented or independently verified',
  capabilities: {
    static_image_landmarks: true,
    dynamic_signs: false,
    video_temporal_inference: false,
    out_of_vocabulary_training: false,
  },
};

export const DROWSINESS_MODEL_MANIFEST_JSON = {
  task: 'driver drowsiness detection',
  status: 'incomplete_no_validated_drowsiness_model',
  person_detector: {
    architecture: 'Ultralytics YOLO11n',
    classes_used: { '0': 'person' },
    runtime: 'ultralytics==8.4.173',
    local_path: 'yolo11n.pt',
    sha256: '0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1',
    license_status: 'The exact local weight provenance and redistribution terms are not established by the repository.',
  },
  drowsiness_model: {
    architecture: null,
    source: null,
    version: null,
    weights: null,
    sha256: null,
    supported_labels: [],
    temporal_window_seconds: null,
    status: 'unavailable',
  },
  face_eye_landmark_model: { source: null, weights: null, status: 'unavailable' },
  age_model: { source: null, weights: null, status: 'unavailable_optional' },
  datasets: [],
  evaluation: {
    drowsiness_metrics: null,
    subject_independent_test: null,
    status: 'not_performed',
  },
};

export const PROJECT_DATASETS: DatasetEntry[] = [
  {
    id: 'coco-2017-car-detection',
    taskNumber: 1,
    taskName: 'Car Colour Detection',
    datasetName: 'MS COCO 2017 (Object Detection)',
    role: 'Pretrained Backbone Dataset',
    modelUsed: 'Ultralytics YOLO11n + HSV Crop-Feature Estimator',
    modelArtifact: 'yolo11n.pt',
    weightsSha256: '0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1',
    labelsCovered: ['person (COCO 0)', 'car (COCO 2)', 'HSV palette: blue, red, white, black, gray, silver, green, yellow, orange, brown, other, unknown'],
    license: 'COCO Terms of Use (CC BY 4.0 annotations) / Ultralytics AGPL-3.0',
    sourceUrl: 'https://cocodataset.org/#home',
    modelUrl: 'https://docs.ultralytics.com/models/yolo11/',
    notes: 'Used to pretrain YOLO11n for vehicle and pedestrian bounding boxes. Car body colour is estimated via a separate non-trained HSV crop-feature heuristic.',
    limitations: 'General COCO detector (not traffic-specific); apparent colour is sensitive to lighting, reflections, and occlusion.',
  },
  {
    id: 'coco-2017-wildlife',
    taskNumber: 2,
    taskName: 'Wildlife Risk Intelligence',
    datasetName: 'MS COCO 2017 (Animal Classes) + Dietary Taxonomy v1.0.0',
    role: 'Bundled Taxonomy / Manifest',
    modelUsed: 'Ultralytics YOLO11n + Deterministic Taxonomy Lookup',
    modelArtifact: 'yolo11n.pt + taxonomy.json',
    weightsSha256: '0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1',
    labelsCovered: ['bird', 'cat (carnivore)', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
    license: 'COCO Terms of Use / Britannica reference citations in taxonomy.json',
    sourceUrl: 'https://cocodataset.org/#home',
    modelUrl: 'https://docs.ultralytics.com/models/yolo11/',
    notes: 'Maps the 10 COCO animal labels to carnivore, non_carnivore, or unknown via versioned taxonomy.json (v1.0.0).',
    limitations: 'Broad COCO classes rather than fine-grained wildlife species; does not cover deer, foxes, wolves, lions, or tigers.',
  },
  {
    id: 'iemocap-superb-voice',
    taskNumber: 3,
    taskName: 'Voice Emotion Detection',
    datasetName: 'IEMOCAP (via SUPERB Emotion Recognition Benchmark)',
    role: 'Pretrained Backbone Dataset',
    modelUsed: 'superb/wav2vec2-base-superb-er',
    modelArtifact: 'models/voice_emotion/pytorch_model.bin (378 MB, rev 441a7599)',
    weightsSha256: null,
    labelsCovered: ['neutral (neu)', 'happy (hap)', 'angry (ang)', 'sad (sad)'],
    license: 'Model weights: Apache-2.0; IEMOCAP dataset requires separate USC SAIL license/access',
    sourceUrl: 'https://sail.usc.edu/iemocap/',
    modelUrl: 'https://huggingface.co/superb/wav2vec2-base-superb-er',
    notes: 'Mono 16 kHz PCM WAV utterance-level speech emotion recognition trained on IEMOCAP via the SUPERB benchmark.',
    limitations: 'English-language acted/scripted benchmark domain; 0.6258 reported accuracy on one SUPERB/IEMOCAP session. Never infers gender or identity.',
  },
  {
    id: 'coco-2017-person',
    taskNumber: 4,
    taskName: 'Person Attribute & Emotion Analysis',
    datasetName: 'MS COCO 2017 (Person Detection)',
    role: 'Pretrained Backbone Dataset',
    modelUsed: 'Ultralytics YOLO11n (person class only)',
    modelArtifact: 'yolo11n.pt',
    weightsSha256: '0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1',
    labelsCovered: ['person (COCO 0)'],
    license: 'COCO Terms of Use / Ultralytics AGPL-3.0',
    sourceUrl: 'https://cocodataset.org/#home',
    modelUrl: 'https://docs.ultralytics.com/models/yolo11/',
    notes: 'Performs person detection only. Optional age and facial-expression datasets/models are intentionally omitted until validated models are provisioned.',
    limitations: 'Nationality and ethnicity are never inferred from appearance. Age and expression analysis remain disabled.',
  },
  {
    id: 'asl-alphabet-landmarks',
    taskNumber: 5,
    taskName: 'Sign Language Detection',
    datasetName: 'Static ASL Fingerspelling Alphabet (21-Point Hand Landmarks)',
    role: 'Pretrained Backbone Dataset',
    modelUsed: 'MediaPipe Hands + AmimulBmeIU/asl-alphabet-classifier (Keras)',
    modelArtifact: 'models/sign_language/asl_landmarks.keras (245,970 bytes)',
    weightsSha256: '9A73CD144A41C2CA92BD4D07ECEC2FE32F5F4DDE5E28C8BCC0B0ADA013502F11',
    labelsCovered: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'],
    license: 'MIT (publisher-declared for weights; underlying raw image dataset not specified by publisher)',
    sourceUrl: 'https://huggingface.co/AmimulBmeIU/asl-alphabet-classifier',
    modelUrl: 'https://huggingface.co/AmimulBmeIU/asl-alphabet-classifier',
    notes: 'Classifies 24 static ASL letters using 21 normalized (x, y, z) hand landmarks from MediaPipe Hands. Enforces configurable time window (09:00–17:00 America/New_York).',
    limitations: 'Excludes motion letters J and Z. Publisher card reports 0.9415 validation accuracy without documenting training dataset or signer splits.',
  },
  {
    id: 'drowsiness-coco-candidates',
    taskNumber: 6,
    taskName: 'Driver Drowsiness Detection',
    datasetName: 'MS COCO 2017 (Person) + Evaluated Candidates: UTA-RLDD & Simuletic',
    role: 'Evaluated External Candidate',
    modelUsed: 'Ultralytics YOLO11n (Person Detection Only; Drowsiness Model Unprovisioned)',
    modelArtifact: 'yolo11n.pt + model_manifest.json',
    weightsSha256: '0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1',
    labelsCovered: ['person (COCO 0)', 'drowsy/sleeping: unavailable (requires licensed temporal video dataset)'],
    license: 'Simuletic sample: CC-BY-4.0; UTA-RLDD derivative: license unverified',
    sourceUrl: 'https://huggingface.co/datasets/Simuletic/Driver_Drowsiness_Monitoring_Detection_Dataset',
    modelUrl: 'https://huggingface.co/datasets/chbh7051/UTA-RLDD_images',
    notes: 'Currently detects people only. Documented external candidates evaluated in README: Simuletic Driver Drowsiness Monitoring Detection Dataset (168 synthetic frames) and UTA-RLDD images.',
    limitations: 'Neither external candidate was provisioned due to synthetic-sample / unverified-license limitations. Sleeping/drowsy counts stay explicitly unavailable.',
  },
];

export interface GeneratedBundleFile {
  id: string;
  filename: string;
  mimeType: string;
  description: string;
  content: string;
}

function escapeCsv(value: string): string {
  if (value.includes(',') || value.includes('"') || value.includes('\n')) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}

export function buildSubmissionFiles(): GeneratedBundleFile[] {
  const csvHeader = [
    'Task #',
    'Module',
    'Dataset Name',
    'Role',
    'Model / Pipeline',
    'Artifact',
    'SHA-256',
    'Labels Covered',
    'License',
    'Dataset URL',
    'Model URL',
    'Notes & Limitations',
  ].join(',');

  const csvRows = PROJECT_DATASETS.map((item) =>
    [
      String(item.taskNumber),
      item.taskName,
      item.datasetName,
      item.role,
      item.modelUsed,
      item.modelArtifact,
      item.weightsSha256 ?? 'Not pinned / external',
      item.labelsCovered.join('; '),
      item.license,
      item.sourceUrl,
      item.modelUrl,
      `${item.notes} Limitations: ${item.limitations}`,
    ]
      .map(escapeCsv)
      .join(','),
  );

  const csvContent = [csvHeader, ...csvRows].join('\n');

  const readmeContent = `# RoadSight Intelligence — Internship Dataset Submission Package

Generated from the **RoadSight Intelligence** AI/ML Internship Project repository.

## Datasets Summary

| Task | Module | Dataset | Model / Weights | Source |
| :--- | :--- | :--- | :--- | :--- |
${PROJECT_DATASETS.map(
  (d) =>
    `| Task ${d.taskNumber} | ${d.taskName} | ${d.datasetName} | ${d.modelUsed} | ${d.sourceUrl} |`,
).join('\n')}

## Detailed Dataset & Provenance Notes

${PROJECT_DATASETS.map(
  (d) => `### Task ${d.taskNumber} — ${d.taskName}
- **Dataset:** ${d.datasetName} (${d.role})
- **Model:** ${d.modelUsed}
- **Artifact:** \`${d.modelArtifact}\`
${d.weightsSha256 ? `- **SHA-256:** \`${d.weightsSha256}\`\n` : ''}- **Labels / Classes:** ${d.labelsCovered.join(', ')}
- **License / Terms:** ${d.license}
- **Dataset Link:** ${d.sourceUrl}
- **Model Link:** ${d.modelUrl}
- **Notes:** ${d.notes}
- **Limitations:** ${d.limitations}
`,
).join('\n')}

## Included Files in This Submission Folder

1. \`datasets_summary.csv\` — Spreadsheet-ready summary of all 6 tasks, datasets, licenses, and URLs.
2. \`datasets_catalog.json\` — Structured machine-readable dataset and model provenance catalog.
3. \`INTERNSHIP_DATASETS_README.md\` — This human-readable submission overview.
4. \`wildlife_taxonomy.json\` — Versioned v1.0.0 COCO animal-to-dietary classification dataset/mapping.
5. \`sign_language_vocabulary.json\` — Supported 24-letter static ASL fingerspelling label schema.
6. \`sign_language_model_manifest.json\` — Checksum-pinned ASL classifier provenance manifest.
7. \`drowsiness_model_manifest.json\` — Driver drowsiness module artifact & dataset status manifest.
`;

  return [
    {
      id: 'datasets-summary-csv',
      filename: 'datasets_summary.csv',
      mimeType: 'text/csv',
      description: 'Spreadsheet-ready table of all 6 tasks, datasets, models, licenses, and links.',
      content: csvContent,
    },
    {
      id: 'datasets-catalog-json',
      filename: 'datasets_catalog.json',
      mimeType: 'application/json',
      description: 'Complete structured JSON catalog with SHA-256 hashes, labels, and limitations.',
      content: JSON.stringify(
        {
          project: 'RoadSight Intelligence — AI/ML Internship Project',
          generated_at: new Date().toISOString(),
          datasets: PROJECT_DATASETS,
        },
        null,
        2,
      ),
    },
    {
      id: 'internship-readme-md',
      filename: 'INTERNSHIP_DATASETS_README.md',
      mimeType: 'text/markdown',
      description: 'Formatted internship portal documentation sheet explaining every dataset.',
      content: readmeContent,
    },
    {
      id: 'wildlife-taxonomy-json',
      filename: 'wildlife_taxonomy.json',
      mimeType: 'application/json',
      description: 'Versioned dietary classification taxonomy (v1.0.0) for Task 2 Wildlife.',
      content: JSON.stringify(WILDLIFE_TAXONOMY_JSON, null, 2),
    },
    {
      id: 'asl-vocabulary-json',
      filename: 'sign_language_vocabulary.json',
      mimeType: 'application/json',
      description: 'Declared 24-letter static ASL alphabet label schema for Task 5.',
      content: JSON.stringify(SIGN_LANGUAGE_VOCABULARY_JSON, null, 2),
    },
    {
      id: 'asl-manifest-json',
      filename: 'sign_language_model_manifest.json',
      mimeType: 'application/json',
      description: 'Pinned checkpoint metadata & provenance manifest for Task 5.',
      content: JSON.stringify(SIGN_LANGUAGE_MODEL_MANIFEST_JSON, null, 2),
    },
    {
      id: 'drowsiness-manifest-json',
      filename: 'drowsiness_model_manifest.json',
      mimeType: 'application/json',
      description: 'Model and dataset status manifest for Task 6 Drowsiness Detection.',
      content: JSON.stringify(DROWSINESS_MODEL_MANIFEST_JSON, null, 2),
    },
  ];
}
