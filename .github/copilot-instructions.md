# RoadSight Intelligence — GitHub Copilot Instructions

## 1. Project rules
- Work only inside the currently opened RoadSight Intelligence repository.
- Maintain one integrated Streamlit application with six independent, testable ML modules.
- Inspect `PROJECT_SPEC.md`, `README.md`, the existing code, and Git status before making changes.
- Reuse the existing project architecture, environment, and working components wherever possible.
- Implement only the task explicitly requested. Do not start future tasks automatically.

## 2. Task sequence
1. Car Colour Detection — existing implementation.
2. Wildlife/Animal Detection — existing implementation.
3. Voice Emotion Detection.
4. Person Analysis.
5. Sign Language Detection.
6. Driver Drowsiness Detection.

Keep each task in its own module. Avoid changing completed modules unless necessary to fix a demonstrated regression or integrate navigation.

## 3. Model and data integrity
- Use documented, real model inference for production predictions.
- Never fabricate detections, metrics, training results, or confidence values.
- Never claim a model has been trained when only pretrained weights were used.
- Document model sources, versions, licenses, supported classes, limitations, and setup steps.
- If weights or dependencies are missing, report the exact issue. Do not silently substitute mock predictions.
- Do not infer nationality or ethnicity from a person's appearance.
- Do not infer voice gender as a reliable fact. Follow the project's explicit consent and eligibility requirements.
- Make limitations and unsupported classes clear in the UI and documentation.

## 4. Before changing code
1. Inspect `git status`, relevant diffs, project specifications, and tests.
2. Run the existing test suite and report the baseline.
3. Identify the files and interfaces relevant to the requested task.
4. Explain the planned changes and any dependency or model requirements.
5. Avoid unrelated refactoring.

## 5. Implementation and testing
- Keep inference logic separate from Streamlit presentation code.
- Validate uploaded images, audio, and video before processing.
- Handle missing/corrupt models, unsupported media, codec failures, and invalid configuration explicitly.
- Add unit and integration tests for the requested task.
- Preserve all existing tests and functionality.
- Run targeted tests first, then the complete test suite.
- Run Python syntax/compile checks and `git diff --check`.
- Report actual test counts and failures; never claim success without command results.

## 6. Permissions and Git safety
- Do not commit or push without explicit approval.
- Do not delete files, overwrite unrelated work, install packages, or download large models without explaining the need and receiving approval.
- Never include secrets, virtual environments, generated media, or unnecessary model weights in Git.
- Never discard user changes to obtain a clean working tree.
- Show the final diff summary before proposing a checkpoint.

## 7. Definition of done
For each task, report:
1. Implemented functionality.
2. Files created or modified.
3. Model and weights actually used.
4. Commands executed and exact test results.
5. Known limitations and manual verification steps.
6. Git status and recommended next action.

Stop after completing the requested task. Wait for approval before starting the next task.