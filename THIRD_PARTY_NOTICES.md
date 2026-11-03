# External dependency notices

The MIT license in this repository applies to project-owned code and
documentation. It does not replace the licenses of separately installed
dependencies, model weights, or external services.

## Super-resolution components

| Component | Relationship to this repository | Upstream license reference |
| --- | --- | --- |
| Real-ESRGAN | Imported as an external dependency; its model weights are obtained separately. | [BSD-3-Clause, copyright 2021 Xintao Wang](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE) |
| BasicSR | External dependency providing the imported `RRDBNet` architecture. | [Apache-2.0, BasicSR Authors](https://github.com/XPixelGroup/BasicSR/blob/master/LICENSE.txt) |

This repository does not vendor these packages or redistribute their
pretrained weights. The model configuration and API calls in the SR wrapper
do not relicense the imported implementation. Verify the license and any
notices of the exact upstream versions and weights you obtain; upstream
default branches can change.

## Other dependencies and services

NumPy, OpenCV, pandas, scikit-learn, scikit-image, PyTorch, torchvision,
OpenPyXL, Pillow, Requests and their transitive dependencies are installed
separately through the requirements files and retain their own licenses.
Their installed distributions provide the applicable license texts.

Qwen and Doubao are optional external API services, not software distributed
with this code. Their service terms and any data permissions must be assessed
separately. Patient images, clinical datasets, expert ratings, and pretrained
weights are not included in this repository or licensed by its MIT file.

## Source provenance

The project owner confirmed authority to license the supplied research
scripts under MIT. Preparation of the public release retained selected
functional scripts, corrected kernels, and adapted execution wrappers;
see the README for the scope and limitations. A scan of the selected Python
files found no additional copyright headers, embedded license declarations,
or copied-from source attributions. That scan is not a complete legal audit
of code provenance or every dependency.
