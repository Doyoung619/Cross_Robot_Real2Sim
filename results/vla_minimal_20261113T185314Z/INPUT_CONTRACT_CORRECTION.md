# Input contract correction

The initial run omitted the official LiberoProcessorStep 180-degree image rotation. Its artifacts are preserved in history/initial_image_contract_bug and are not used as final transfer evidence. The corrected pipeline applies the documented rotation to both offline training and evaluation images, renders 256px inputs, and uses previously unused reserved scenes demo_45/demo_46 for final evaluation. Source selection and learning hyperparameters remain unchanged; no B trajectory enters training.
