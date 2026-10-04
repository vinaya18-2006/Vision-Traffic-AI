# AI/ML Internship Integrated Application Specification

## 1. Project Purpose and Scope

Build one Python application with a Streamlit graphical interface that exposes six independently testable ML modules. The Streamlit layer coordinates user input and displays results; it does not contain task-specific model or inference logic. Each module must use real inference from documented, reproducibly obtainable model weights. Missing or invalid weights are an explicit error state, never a reason to return a placeholder prediction.

This document specifies the intended system. It does not claim that models, datasets, weights, implementations, or dependencies currently exist. A task is incomplete until its Definition of Done is met with real model inference and its documented tests pass.

## 2. Intended Future Repository Layout

This is a target layout for future work, not a request to create it now.

```text
AI-ML-Internship-Project/
├── app.py
├── PROJECT_SPEC.md
├── README.md
├── requirements.txt
├── .gitignore
├── models/
├── modules/
│   ├── car_detection/
│   ├── animal_detection/
│   ├── voice_emotion/
│   ├── person_analysis/
│   ├── sign_language/
│   └── drowsiness/
├── assets/
├── notebooks/
└── tests/
```

Each module owns its task-specific preprocessing, model adapter/loader, inference, post-processing, and module-level tests. Shared utilities may handle validated media decoding, configuration, common result types, and visualization, but must not couple one task's model to another task. Keep inference callable without importing Streamlit.

## 3. Shared Contracts and Engineering Requirements

- Define stable input and result contracts for each module. Results include the source item/frame identity, model/version metadata, detections or predictions, confidence when meaningful, and any warnings. Do not imply that a confidence score is calibrated probability unless calibration has been measured.
- Keep uploaded-media handling, UI state, and presentation in the Streamlit layer; call module APIs for inference. The UI must label the selected task and input, and distinguish processed media from the original.
- Configure model paths, thresholds, class maps, and task-specific options through validated configuration/environment variables or documented user settings. Do not hardcode machine-specific paths or secrets.
- On startup or task selection, validate that required model files exist, can be loaded, and match the expected format/version. Report the exact missing or invalid artifact and documented acquisition route. Do not silently substitute another model, hand-coded logic, or mock output.
- Dataset and model choices must be documented with source, version, license/terms, intended use, limitations, and reproducible acquisition/training steps. Confirm redistribution and commercial-use terms before committing artifacts.
- Keep large weights and datasets out of Git when appropriate. Ignore `.venv`, generated media, caches, secrets, and local configuration. Do not commit API keys or credentials. Track small configuration/manifests and document how to obtain omitted artifacts.
- Pin compatible runtime dependencies and document Windows/VS Code setup, virtual-environment activation, install, run, and test commands. Dependency installation is a setup step, not part of this specification.
- Each module must be independently importable, callable, and testable. Tests must cover valid inputs, malformed/unsupported inputs, missing/corrupt weights, confidence/threshold behavior, and media cleanup. Module tests must not require launching the full GUI unless testing integration.
- Test doubles may be used only to test plumbing and error handling; they must be clearly confined to tests and must never be surfaced as model predictions or used as a production fallback. Model/inference acceptance tests use actual documented weights.
- Image and video outputs must preserve the source and clearly identify any annotated copy. Do not overwrite user uploads. Avoid retaining uploaded media beyond the documented session lifecycle.
- Any limitation that affects reliability (occlusion, lighting, language/vocabulary coverage, recording quality, demographic/model bias, or temporal context) must be visible in documentation and, where material, in the UI.

## 4. Task Specifications

### Task 1 — Car Colour Detection

1. **Purpose:** Detect cars and people in traffic-signal scenes, count both, and classify the apparent body colour of each detected car.
2. **User inputs:** One image or a video file; optional validated confidence, colour-confidence, and processing settings.
3. **Supported formats:** Common still images (JPEG, PNG, WebP where decoder support is verified) and video (MP4, AVI, MOV where the installed OpenCV backend supports the codec). Display supported formats from actual backend capability.
4. **Expected outputs:** Annotated image/video, car count, people count, and per-car class, box, colour label, and colour confidence or an explicit unknown/low-confidence label.
5. **ML/computer-vision approach:** Run object detection first; filter vehicle detections to cars and separately retain person detections. Classify each car's colour from its vehicle crop. Keep vehicle detection and colour classification as logically separate stages and independently testable components.
6. **Recommended model architecture:** A pretrained object detector with car/person classes (YOLO-family detector is a candidate), followed by a separate crop-based colour classifier or documented robust colour-feature method. Choose only after validating domain performance; do not make colour-specific detector classes the default.
7. **Pretrained/custom-trained expectation:** Start with a suitably licensed pretrained detector. Use a pretrained colour model only if it covers the required classes and domain; otherwise train/fine-tune a separate colour classifier. Fine-tuning the detector is needed only if evaluation shows the target traffic domain/classes are not adequately supported.
8. **Dataset requirements:** Document a licensed, representative traffic/car dataset for detector validation/training and labeled car crops for colour classification. Include varied lighting, weather, viewpoints, occlusion, and paint colours. Split by source/video to avoid leakage.
9. **Annotation requirements:** Detector boxes/classes for cars and people; separate car-crop colour labels with a defined palette and an unknown/ambiguous policy. Record annotation guidance and inter-annotator quality checks. Do not label blue cars as a separate detector class unless measured evidence justifies that design.
10. **Preprocessing:** Decode and validate media, convert channel order correctly, resize/normalize as required by the selected detector, then crop car boxes with clipping and minimum-size checks. Define the colour classifier's colour-space/normalization steps and account for illumination where supported.
11. **Inference pipeline:** Validate weights and media; detect objects; filter and threshold detections; classify each car crop; apply configured per-stage thresholds; create structured detections and annotated copies. For video, process frames and optionally track objects using a documented tracker.
12. **Post-processing:** Apply documented NMS/threshold rules and stable class/colour names. Draw blue-car boxes in **RED** and cars of every other known colour in **BLUE**. Choose a documented neutral style for unknown colour. Draw people distinctly from cars. Do not turn low-confidence classifications into confident labels.
13. **GUI requirements:** Provide image/video input, processing controls, annotated output, per-car colour results, car count, people count, and clear model/loading/error states. Show whether counts are for a still image, current frame, or a video summary.
14. **Image requirements:** Annotate every accepted detected car/person; make boxes and labels legible without obscuring the subject. For a still image, count each retained detection once.
15. **Video requirements:** Display an annotated video or a clearly selected annotated frame sequence with timing/progress. Handle decode failure, frame limits, and cleanup. Do not imply real-time performance unless measured.
16. **Counting requirements:** Count unique retained car/person detections per still image. For video, report per-frame counts and/or tracked unique-object totals with the chosen policy and tracker limitations explicit; never sum every frame as a unique-object count.
17. **Error handling:** Report unsupported/corrupt media, no detections, missing/invalid weights, incompatible model classes, unusably small crops, and decode/write failures. Empty results are valid only after real inference ran successfully.
18. **Testing requirements:** Unit-test crop bounds, colour mapping, annotation rules (blue car red; other car blue), and count policies. Use real model inference tests for representative stills and video clips, plus missing/corrupt-model, empty-detection, codec, and malformed-file cases.
19. **Model-weight requirements:** Record detector and colour model names, source/version, class map, preprocessing, license, file format, configurable path, checksum, and exact acquisition/training instructions. Both models must load actual weights before inference.
20. **Definition of Done:** Real car/person detection and separate colour classification work on documented image/video examples; required colours and counts are displayed correctly; blue cars are red-boxed and other known colours blue-boxed; errors and limitations are handled; unit/inference/integration tests, documentation, and reproducible weight acquisition are complete.

**Model strategy (A-H):** A. Pretrained car/person detector where suitable. B. Fine-tune detector only if documented evaluation requires it; train a separate colour classifier if no suitable pretrained one exists. C. Use licensed traffic imagery and labeled car-colour crops. D. Store detector and classifier weights separately with manifests/checksums. E. Apply each model's documented resize/normalization and crop-specific colour preprocessing. F. Detect cars/people, then classify each car crop. G. Threshold/NMS, map colour, count according to image/video policy, and annotate with the required colours. H. Streamlit selects input/settings and renders structured outputs; no inference logic in UI.

### Task 2 — Animal Detection

1. **Purpose:** Detect and classify multiple supported animal species in images/videos, highlight carnivorous species, and report their count.
2. **User inputs:** One image or video; optional confidence threshold and documented species/category settings.
3. **Supported formats:** Common still-image and video formats only where the configured decoder supports them; validate and display accepted types.
4. **Expected outputs:** Annotated media, per-animal species/box/confidence/category, total supported-animal count, and carnivorous-animal count prominently visible in the GUI or a result dialog.
5. **ML/computer-vision approach:** Use an object detector/classifier for species labels, then map supported species to dietary categories using a maintained, sourced mapping. The mapping is a separate, versioned data/config component, not arbitrary inference logic.
6. **Recommended model architecture:** Pretrained multi-class object detector for animal localization/species; fine-tuned detector or second-stage classifier if the required species are absent or inaccurate in the chosen model.
7. **Pretrained/custom-trained expectation:** Prefer a real pretrained model when its documented class vocabulary covers the requested species adequately. Custom training/fine-tuning is required to add unsupported species, with appropriate labeled data.
8. **Dataset requirements:** Use licensed, species-labeled images/videos spanning realistic poses, scales, habitats, lighting, and occlusion. Document class coverage and imbalance. Gather authoritative, versioned dietary-category references for the supported species.
9. **Annotation requirements:** Bounding boxes and canonical species names; map synonyms to a stable taxonomy. Define treatment for ambiguous/unknown species and species-level mapping confidence. Document category source and review process.
10. **Preprocessing:** Validate/decode media, normalize as required by the model, preserve spatial transforms for boxes, and convert predicted labels to canonical taxonomy keys before category lookup.
11. **Inference pipeline:** Load actual weights; detect/classify animals; threshold and normalize labels; look up each supported species in the configured carnivore mapping; produce structured detections and annotated media.
12. **Post-processing:** Draw carnivorous animals with **RED** boxes and label them; use a distinct documented style for other/unknown categories. Count only supported detections whose mapping is known to be carnivorous. Never infer diet from appearance.
13. **GUI requirements:** Accept image/video, show processed output, species and category labels, total detections and an immediately visible carnivorous-animal count. Expose the mapping version/source and unknown-category warning accessibly.
14. **Image requirements:** Annotate individual detected animals and show a single-image count; do not claim a species absent from the model vocabulary.
15. **Video requirements:** Annotate processed frames/video, report progress and the selected count basis, and clean temporary media. Do not equate repeated frame detections with unique animals.
16. **Counting requirements:** Count per-image detections for images. For video, distinguish per-frame totals from unique tracked animals; use a documented tracker if reporting unique individuals. Carnivore count is derived only from detections with a known mapping.
17. **Error handling:** Report missing model/mapping, unsupported species, unknown category, invalid media, decoder errors, and inference failure. Unknown mapping must not silently become herbivore or carnivore.
18. **Testing requirements:** Test taxonomy normalization, species mapping, unknown species behavior, red carnivore annotation, and count logic. Run real-weight image/video tests and failure tests for missing mappings/weights, unsupported labels, and bad media.
19. **Model-weight requirements:** Document species vocabulary, source/version/license, class map, preprocessing, configurable weights path, checksum, and acquisition/fine-tuning process. Version the species-to-category mapping separately, with citations and review date.
20. **Definition of Done:** Real inference detects supported species in image/video; a documented mapping correctly determines carnivore status; red boxes/counts and other results are clear; unknowns/errors are explicit; tests, limitations, mapping provenance, and reproducible model acquisition are documented.

**Model strategy (A-H):** A. Use a suitable pretrained multi-species detector if its class list suffices. B. Fine-tune/train only to provide verified additional species coverage. C. Use licensed labeled species data and independently sourced category references. D. Version actual detector weights and the category mapping separately. E. Apply model-specific image/video transforms and canonicalize taxonomy labels. F. Detect/classify, then perform deterministic mapping lookup. G. Threshold detections, distinguish unknowns, draw carnivores red, and count under the stated policy. H. GUI shows annotations, counts, and mapping provenance without implementing species inference.

### Task 3 — Emotion Detection Through Voice

1. **Purpose:** Estimate an audio emotion class from supported speech using a real trained speech-emotion model; output is a model classification, not a definitive statement of a speaker's internal state.
2. **User inputs:** Uploaded audio/voice note; optionally recorded voice through a browser-supported microphone component with permission. If eligibility is restricted to female voices, use explicit user-provided eligibility/consent input where genuinely required; never infer gender from voice.
3. **Supported formats:** WAV and any compressed formats (for example MP3/M4A) only if the documented decoder can decode them. Validate sample rate, channels, duration, and file size.
4. **Expected outputs:** Predicted supported emotion label, score/confidence where meaningful, optional per-segment results, and warnings for silence, poor quality, out-of-domain input, or low confidence.
5. **ML/computer-vision approach:** Speech-emotion recognition from audio features or learned waveform/spectrogram representations. Do not use a hidden gender classifier. Emotion labels describe model categories and are subject to dataset and cultural limitations.
6. **Recommended model architecture:** A documented pretrained audio/speech encoder with an emotion classification head, or a spectrogram-based CNN/CRNN; choose based on licensing, supported labels, and validation. Fine-tuned classifier must match the declared emotion vocabulary.
7. **Pretrained/custom-trained expectation:** A pretrained SER model may be used if it genuinely supports the required labels and language/domain. Fine-tuning is expected when task requirements, labels, language, or recording conditions are not covered. Record any internship-specific training requirement explicitly.
8. **Dataset requirements:** Use licensed, consented speech-emotion datasets with speaker-independent train/validation/test splits. Document language, demographic coverage, recording conditions, label definitions, imbalance, and known limitations. Never use test speakers in training.
9. **Annotation requirements:** Emotion labels must have a documented schema and annotation protocol; record annotator agreement and ambiguous/neutral cases. Any gender/eligibility rule must be an explicit user-provided field and have a justified, documented purpose, not a model inference.
10. **Preprocessing:** Decode audio, resample to model rate, convert channels as specified, normalize only as documented, segment/pad within allowed duration, and handle silence/clipping. Do not discard or retain audio without a documented policy.
11. **Inference pipeline:** Validate file/recording and weights; extract model-required waveform/features; run trained emotion classifier; map outputs to the declared vocabulary; apply a documented low-confidence/unknown rule; return result and model metadata.
12. **Post-processing:** Aggregate segment predictions only by a documented policy. Show scores as model scores, not calibrated probabilities unless validated. Do not infer gender, identity, nationality, or other unrelated attributes from the voice.
13. **GUI requirements:** Upload and, where technically available, record audio; show recording permission/state, playback or duration, predicted label, confidence/uncertainty, and actionable validation errors. Clearly communicate supported labels and model limitations.
14. **Image requirements:** Not applicable. Do not accept an image as a voice input; return a clear unsupported-input message.
15. **Video requirements:** Video is not a required input. If later added, audio extraction and synchronization must be explicitly specified and tested; do not infer emotion from video as a substitute for audio.
16. **Counting requirements:** No object counting. If segmenting an audio file, show segment-level and aggregate results separately and do not count segments as people or independent speakers.
17. **Error handling:** Handle absent microphone permission, unavailable recording component, unsupported/corrupt audio, silence, excessive duration/size, wrong sample rate when conversion is unavailable, missing weights, and low-confidence/out-of-domain input. Never fabricate a fallback label.
18. **Testing requirements:** Unit-test decoding, resampling/feature shapes, segmentation, label mapping, and errors. Run real-weight tests on held-out audio, plus upload and microphone GUI smoke tests where browser support exists. Include silence, noise, clipping, unsupported format, and missing-model cases.
19. **Model-weight requirements:** Document model/encoder and classifier versions, label vocabulary, language, dataset/training provenance, license, preprocessing, calibration status, checksum, configurable path, and exact acquisition/fine-tuning procedure.
20. **Definition of Done:** Real audio inference runs for supported uploads and supported recording input; labels/scores and limitations are presented accurately; no gender is inferred; invalid/missing/low-quality cases are handled; real-weight tests, GUI smoke tests, documentation, and reproducible setup are complete.

**Model strategy (A-H):** A. Use a verified pretrained SER encoder/classifier if its label/language scope matches. B. Fine-tune a classification head or model where validation and internship requirements call for it. C. Use consented, licensed emotion-labeled speech with speaker-independent splits. D. Document and configure real encoder/classifier weights. E. Decode/resample/segment and extract features exactly as training expects. F. Run only the trained emotion inference pipeline. G. Apply documented segment aggregation and uncertainty rules; no gender inference. H. UI handles upload/recording and renders the model's bounded result and limitations.

### Task 4 — Person Attribute & Emotion Analysis

1. **Purpose:** Detect people and, only where suitable real models exist, report bounded non-sensitive visual attributes, coarse age estimates, and facial-expression classifier results.
2. **User inputs:** Image or video; optional explicit user-provided nationality/region value only if a documented downstream branch truly requires it. Never infer nationality or ethnicity from a person's face, clothing, skin tone, or appearance.
3. **Supported formats:** Common images and supported video codecs, validated through the selected decoder.
4. **Expected outputs:** Person boxes; optional coarse age range with uncertainty; optional non-sensitive visible attributes supported by a validated model; optional facial-expression category/score clearly described as an expression estimate. Any branch based on nationality/region uses only the explicit input and identifies it as user-provided.
5. **ML/computer-vision approach:** Person detector followed by separate, optional face/attribute/expression models on detected crops. Keep each model's scope explicit. Facial-expression classification is not a reliable measurement of internal emotional state.
6. **Recommended model architecture:** Pretrained person detector plus independently evaluated face landmark/crop, coarse age, non-sensitive attribute, and facial-expression models as justified. Use separate heads/models where task labels and licensing differ.
7. **Pretrained/custom-trained expectation:** Pretrained components are acceptable only when class coverage, intended use, license, and evaluation support the feature. Custom training/fine-tuning is necessary for unsupported labels or requirements; omit features for which no suitable real model exists. No appearance-based nationality/ethnicity classifier may be built or integrated.
8. **Dataset requirements:** Use appropriately licensed, consented datasets with documented coverage and limitations for each model. For attributes and expression, define label scope and evaluate across relevant lighting, pose, age, and demographic subgroups without making identity/nationality claims.
9. **Annotation requirements:** Person boxes; task-specific labels only for supported features. Age labels should be coarse ranges, not precise age claims. Expression labels need an annotation protocol and must be framed as visible-expression categories. No nationality or ethnicity labels from appearance.
10. **Preprocessing:** Detect people; crop only as required by downstream models; apply documented face/landmark, resize, normalization, and quality checks. Avoid unnecessary retention of biometric imagery. Explicit user input is validated independently of image processing.
11. **Inference pipeline:** Load only available, documented model weights; detect people; run enabled, supported model stages on appropriate crops; condition any non-sensitive downstream processing on explicit user input if required; produce per-person results with provenance and uncertainty.
12. **Post-processing:** Bound age to coarse ranges and expose uncertainty. Use “expression estimate”/equivalent language, not an assertion of true emotion. Suppress unsupported or low-quality attributes. Branch only on validated explicit nationality/region input when required; never derive it from appearance.
13. **GUI requirements:** Provide media input, per-person annotations and results, optional feature controls, and an explicit user-input field only when a downstream branch needs it. Explain whether a value is user-provided or model-estimated and show missing-model/uncertainty notices.
14. **Image requirements:** Display boxes and only the enabled/supported per-person analyses; report no-detection and low-quality cases accurately.
15. **Video requirements:** Process supported frames, show annotated output/progress, and avoid storing frames beyond the documented session. Report whether results are per-frame or temporally aggregated.
16. **Counting requirements:** Count detected persons per still image. For video, clearly distinguish per-frame detections from tracked unique people; do not sum repeated detections as distinct people.
17. **Error handling:** Handle missing model components individually, invalid media, faces/persons too small or occluded, low confidence, and absent/invalid explicit branch input. If an optional model is unavailable, disable that feature with a clear message rather than returning a guess.
18. **Testing requirements:** Test person/crop mapping, coarse age formatting, uncertainty/quality suppression, explicit-input validation and conditional branches, and prohibited appearance-to-nationality behavior. Run real-weight image/video tests and missing-component, low-quality, and GUI smoke tests.
19. **Model-weight requirements:** Document each component's exact task, architecture, source/version, license, training data, label schema, evaluation/limitations, preprocessing, checksum, configurable path, and acquisition. Do not bundle or claim unavailable component weights.
20. **Definition of Done:** Real person detection and only documented, supported optional analyses work; nationality/ethnicity are never inferred from appearance; any conditional logic uses explicit user input; uncertainty, limitations, and errors are clear; tests, GUI integration, documentation, and reproducible acquisition are complete.

**Model strategy (A-H):** A. Use independently justified pretrained components for detection and each optional analysis. B. Fine-tune only for a defined, safe label set and measured need; exclude appearance-based nationality/ethnicity. C. Use licensed, consented datasets with clear labels and subgroup limitations. D. Keep weights/manifests per component and omit unsupported optional models. E. Apply component-specific person/face crop transforms and validate explicit inputs separately. F. Detect people, then invoke only enabled supported models and explicit-input branches. G. Bound/suppress uncertain outputs and describe expression as an estimate. H. UI distinguishes user-entered values from model outputs and never silently substitutes missing models.

### Task 5 — Sign Language Detection

1. **Purpose:** Recognize a declared, finite vocabulary of sign-language gestures or words using a real trained model; it is not an unrestricted sign-language translator.
2. **User inputs:** Uploaded image and, where camera/browser support permits, uploaded video or real-time webcam stream. Optional language/vocabulary selection only among documented supported models.
3. **Supported formats:** Common still images and supported video formats; webcam access requires browser permission and a technically supported Streamlit component/backend.
4. **Expected outputs:** Supported gesture/word label(s), confidence/uncertainty, annotated image or video/frame results, and an explicit unknown/out-of-vocabulary result when appropriate. Continuous sentence translation is out of scope unless separately trained and specified.
5. **ML/computer-vision approach:** Image gesture recognition can use hand detection/landmarks plus a trained classifier; temporal signs require a sequence model over frames/landmarks. Select the approach based on the declared vocabulary and static/dynamic sign coverage.
6. **Recommended model architecture:** Hand/pose landmark extractor with a CNN/MLP for static signs or temporal CNN/RNN/Transformer for dynamic sequences, or a validated end-to-end model. The architecture must match whether labels are static gestures, isolated words, or continuous signing.
7. **Pretrained/custom-trained expectation:** A pretrained model is acceptable only if its sign language, vocabulary, signer/domain, and input mode match requirements. Otherwise custom training/fine-tuning is required. A model trained on a limited vocabulary must never be represented as arbitrary sign-language understanding.
8. **Dataset requirements:** Use licensed, consented, signer-diverse data for the named sign language and exact vocabulary; for dynamic signs include temporally labeled sequences. Document signer-independent splits, vocabulary coverage, capture conditions, and limitations.
9. **Annotation requirements:** Specify sign-language variant, vocabulary/label definitions, signer consent, sequence boundaries for video, and ambiguous/out-of-vocabulary cases. Record whether samples are isolated signs or continuous signing.
10. **Preprocessing:** Validate image/video, detect/normalize hands or landmarks as specified, maintain temporal order and frame rate, and create model-sized windows with documented padding/missing-hand handling.
11. **Inference pipeline:** Check configured time-window authorization first; load real vocabulary-matched weights; process an image or temporal window; classify only declared labels; apply uncertainty/unknown threshold; return labels with timing and model/vocabulary metadata.
12. **Post-processing:** For video, use only the trained sequence/segmentation policy to combine frame evidence. Do not turn frame-by-frame guesses into a sentence. Preserve unknown/uncertain results and label each supported output as a recognition result, not a translation beyond model scope.
13. **GUI requirements:** Provide image upload and feasible video/webcam input; show supported sign language, model vocabulary, results/annotations, confidence, time-window status, and clear unavailable/denied states. Real-time controls must not imply unsupported language coverage.
14. **Image requirements:** Classify only gestures represented by the image model and vocabulary. Show a clear no-hand/poor-quality/unknown result rather than guessing.
15. **Video requirements:** Support temporal processing for dynamic signs when weights were trained for it. Show processing status and time-aligned predictions. Webcam operation is optional until browser permission, component support, and performance are validated.
16. **Counting requirements:** No people/object count is required. Do not count video frames as recognized signs; count sign events only when a validated temporal segmentation model provides event boundaries.
17. **Error handling:** Report denied camera permission, unsupported input, no visible hands, inadequate frame/window length, out-of-vocabulary gesture, closed operating window, and missing/incompatible weights. No fallback labels.
18. **Testing requirements:** Unit-test window authorization including time-zone and boundary behavior, preprocessing/windowing, vocabulary mapping, and unknown handling. Real-weight tests cover representative image and temporal examples. Add image/video/webcam GUI smoke tests where feasible and missing-model/permission tests.
19. **Model-weight requirements:** Document sign language/variant, exact vocabulary, static/dynamic capability, architecture, source/training data, license, preprocessing, checksum, configurable path, and reproducible training/acquisition. Weights and vocabulary metadata must be version-matched.
20. **Definition of Done:** Real model recognizes its documented finite vocabulary from supported image input and feasible video input; the operating time window is enforced and tested; unknown signs are not guessed; GUI, limitations, real-weight tests, documentation, and reproducible model acquisition are complete.

**Model strategy (A-H):** A. Use a matching pretrained sign model only if its language and vocabulary are verified. B. Train/fine-tune for the required vocabulary and temporal capability otherwise. C. Use consented, licensed, signer-diverse labeled image/sequence data. D. Version actual weights with the vocabulary and sign-language metadata. E. Normalize hands/frames and temporal windows as training requires. F. Enforce configured time authorization, then classify only supported inputs. G. Apply temporal aggregation/event segmentation only if learned and validated; preserve unknowns. H. UI discloses vocabulary, time access, permissions, and whether output is isolated gesture recognition or broader translation.

### Task 6 — Drowsiness Detection

1. **Purpose:** Detect and count people in a vehicle scene, identify supported visible signs of sleeping/drowsiness, mark such people with red boxes, and optionally display coarse age estimates when a suitable model exists.
2. **User inputs:** Image or video of a vehicle/interior; optional documented thresholds and temporal-window settings.
3. **Supported formats:** Common still images and video codecs supported by the decoder; camera/live input is not required unless separately implemented and validated.
4. **Expected outputs:** Annotated image/video, person count, sleeping/drowsy count, per-person status/score and evidence window where applicable, and optional coarse age estimate with uncertainty. Do not display an age when no suitable real model is configured.
5. **ML/computer-vision approach:** Detect people and relevant face/eye/head/posture cues. Drowsiness is a temporal state: video should use a trained temporal classifier or validated temporal rule/model over observable signals. A single image cannot establish sustained drowsiness; any static-image result must be limited to visible sleep-like evidence and disclose this limitation.
6. **Recommended model architecture:** Pretrained person detector plus validated face/landmark/eye detector and a temporal classifier for video (or a documented trained drowsiness detector). Optional independent coarse age model. Avoid relying on a single static eye-closure frame as proof of drowsiness.
7. **Pretrained/custom-trained expectation:** Reuse pretrained person/face components where appropriate. Use a real drowsiness model trained/fine-tuned for the required temporal labels and target camera conditions. Age estimation is optional and must be omitted unless a suitable, evaluated, licensed model is available.
8. **Dataset requirements:** Use licensed, consented, labeled driver/passenger image/video data with temporal alert/drowsy/sleep labels, subject-independent splits, varied illumination/glasses/poses, and clear operational limitations. Age model datasets require separate provenance and evaluation.
9. **Annotation requirements:** Person boxes; per-frame face/eye/posture cues where used; temporal state/window labels for drowsiness; optional coarse age range labels. Define ambiguous, occluded, and not-visible cases. Do not use precise age or infer unrelated identity attributes.
10. **Preprocessing:** Decode frames in order; detect people and relevant crops; preserve timestamps/frame rate; normalize crops/landmarks as model requires; use documented temporal windows and missing/occluded-cue handling. Static images must not be treated as a temporal sequence.
11. **Inference pipeline:** Load real detector and drowsiness weights; detect people; associate relevant face/cue data to each person; process the required temporal window for video; optionally run a real age model; return person-level status, score, time window, and output metadata.
12. **Post-processing:** Mark each supported sleeping/drowsy person with **RED** boxes; distinguish alert, drowsy, sleeping, and unknown only if the model defines those labels. Apply temporal smoothing only if documented and tested. Age is a coarse optional estimate with uncertainty, never a required fabricated field.
13. **GUI requirements:** Accept image/video, show annotated output, detected-person count, sleeping/drowsy count, and optional age estimates where available. Explain static-image limits, temporal window, model availability, and uncertainty. Do not market the result as a substitute for safe driving or medical assessment.
14. **Image requirements:** Detect/count people and mark a person red only when the configured real image model supports the relevant sleep-like visual class; prominently disclose that a still image cannot verify sustained drowsiness. Do not claim temporal certainty from one frame.
15. **Video requirements:** Preserve frame timing and process temporal windows; render annotated video/frames with progress. Report the aggregation window and whether each status is per-frame or window-level. Handle short clips and frame gaps explicitly.
16. **Counting requirements:** Count people per image. For video, report per-frame counts and/or tracked unique people separately. Count sleeping/drowsy people using the documented per-frame/window policy; do not sum the same person across frames as multiple people.
17. **Error handling:** Report missing/incompatible weights, invalid video, insufficient temporal duration, low frame rate, occlusion, no visible face/cues, uncertain state, unsupported codec, and unavailable age model. Use unknown/unavailable, not a guessed state or age.
18. **Testing requirements:** Unit-test person association, temporal-window boundaries, state aggregation, red-box rules, and count semantics. Real-weight image/video tests cover multiple occupants, occlusion, short clips, and varied lighting. Test missing models, optional age disabled, and GUI error states.
19. **Model-weight requirements:** Document detector and drowsiness model sources/versions, temporal labels/window, training/evaluation data, license, preprocessing, checksum, and configurable paths. Document optional age model separately; no weight means no age estimate.
20. **Definition of Done:** Real person detection/counting and documented drowsiness inference work; video uses sufficient temporal evidence; red boxes and counts follow the stated policy; still-image and age limitations are explicit; failures are safe and visible; real-weight tests, GUI integration, documentation, and reproducible acquisition are complete.

**Model strategy (A-H):** A. Reuse suitable pretrained person/face components; select a validated drowsiness model matching image or temporal use. B. Fine-tune/train temporal drowsiness capability if no suitable model meets the defined task; age remains optional. C. Use consented, licensed, temporally labeled data with subject-independent evaluation. D. Document separate real weights for detection, drowsiness, and optional age. E. Preserve frame timing and model-specific crops/windows. F. Detect/associate people, infer state over valid windows, and optionally estimate coarse age. G. Apply only validated temporal aggregation; use red boxes for supported drowsy/sleeping status and explicit unknowns otherwise. H. UI separates counts/status/optional ages and discloses temporal and safety limitations.

## 5. Cross-Project Testing and Acceptance

- **Unit tests:** Pure preprocessing, validation, taxonomy/label maps, thresholding, box rendering rules, aggregation, and configuration behavior for each module.
- **Model/inference tests:** Run each module with its actual documented weights and representative held-out examples. Record expected output schema and quality acceptance criteria; do not replace real inference tests with mocked predictions.
- **Integration tests:** Exercise each module API independently, then exercise the Streamlit-to-module boundary and shared media lifecycle. Verify a missing model yields an explicit unavailable/error state.
- **GUI smoke tests:** Launch the integrated app; verify navigation, input validation, results, loading/error messages, and that each task can be selected without loading unrelated task models.
- **Image tests:** Cover valid, malformed, unsupported, empty-scene, low-quality, and multiple-object images for each applicable task.
- **Video tests:** Cover valid short clips, decode failures, insufficient duration, frame processing, annotation output, count semantics, temporal-window behavior, and cleanup for applicable tasks.
- **Failure cases:** Missing/corrupt/incompatible weights, unavailable optional components, invalid configuration, low confidence, unknown classes, unsupported formats/codecs, permission denial, and resource-limit errors must fail explicitly and must never produce a fabricated result.
- **Reproducibility:** Document environment creation, dependency installation, model/dataset acquisition, configuration, app launch, and test commands for Windows in VS Code. Record dependency and artifact versions/checksums.
- **Project acceptance:** The integrated app remains one application with six separable modules; inference is outside UI code; real weights are required for predictions; model paths are configurable; setup/tests/docs are repeatable; no secrets, virtual environment, or inappropriate large artifacts are committed.
