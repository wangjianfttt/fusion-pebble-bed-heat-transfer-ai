Dear Editor,

We submit the original research article entitled **"Transient conjugate heat transfer in internally heated ceramic pebble beds: pore-resolved reference fields and graph–Transformer prediction"** for consideration in the *International Journal of Heat and Mass Transfer*.

The manuscript addresses pore-scale heat transfer in a solid-breeder fusion blanket, where helium advection, ceramic conduction, internal heating, conjugate fluid–solid exchange and wall cooling act simultaneously. The reference database contains 60 three-dimensional steady OpenFOAM states and 12 fixed-hydrodynamic thermal-step trajectories. Complete operating conditions and endpoint pairs are withheld from model fitting.

Classical response surfaces and dynamic mode decomposition are compared with coordinate PINNs, graph–Transformer models and diffusion-style temperature refinement using common temperature, wall-heat and finite-volume energy measures. A second spherical-pebble packing is calculated at nine matched conditions. Its outlet and maximum-solid temperatures change by less than 0.67%, whereas pressure drop changes by 14.7–18.0%, separating integral thermal robustness from hydraulic sensitivity. Independent HELOKA Nusselt-number and 1-mm fixed-bed pressure-gradient comparisons give mean and median absolute relative errors of 3.87% and 3.71%, respectively.

The work links resolved transport mechanisms, complete-condition generalization and reduced thermal prediction. Its limits are explicit: the transient surrogate concerns post-adjustment thermal evolution with a prescribed hydrodynamic field, and no full-domain or fully coupled startup accuracy is claimed where the corresponding numerical checks did not pass.

This manuscript has not been published previously and is not under consideration by another journal. All authors have approved the manuscript and agree with its submission. The authors declare no competing interests. Data provenance, model inputs and reproducible scripts are described in the manuscript. The public repository at https://github.com/wangjianfttt/fusion-pebble-bed-heat-transfer-ai contains the code, parameter tables, processed figure data and a compact reproduction archive under explicit software and data licences. The final validation-selected predictions and figure records will be added to the same repository, and a versioned Zenodo DOI will be included before submission. The larger decomposed OpenFOAM fields are retained in the institutional archive and are available from the corresponding author on reasonable request.

Thank you for considering this work.

Sincerely,

Jian Wang  
Corresponding author  
Anhui University of Science and Technology  
Institute of Plasma Physics, Hefei Institutes of Physical Science, Chinese Academy of Sciences  
Email: wjfttt@mail.ustc.edu.cn
