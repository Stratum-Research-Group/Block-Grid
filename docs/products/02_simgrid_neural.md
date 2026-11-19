# Product 2: SimGrid Neural — High-Fidelity Multi-Physics PDE Simulator

## 1. Executive Summary
**SimGrid Neural** is an accelerated neural surrogate simulator for continuous physical phenomena (aerodynamics, computational fluid dynamics, weather forecasting, and semiconductor thermal transport). It operates directly on physical grid discretizations where each spatial cell holds full physical tensor fields (stress tensors, velocity gradients, vorticity).

---

## 2. The Core Problem It Solves
Traditional numerical solvers (Navier-Stokes, Maxwell equations) require solving massive linear systems across millions of mesh points, taking hours to compute a single second of simulation. Existing neural operators (FNO, DeepONet, ConvNets):
- Either require local finite-difference stencils (slow global pressure wave propagation).
- Or flatten the physics tensors into unstructured scalars, failing to respect rotational equivariance and continuum mechanics invariants.

---

## 3. How BlockGrid Powers SimGrid Neural
1. **Physical Tensors Kept Intact**:
   Each grid site represents a spatial domain holding a $3 \times 3$ or $8 \times 8$ tensor block ($P \times Q \times F$) representing the stress/strain state.
2. **Two-Sided Frame Transformations ($A_k Z B_k^T$)**:
   Physical frame transformations (rotations, coordinate changes, skew-symmetric vorticity decompositions) are natively executed via BlockGrid's inner operators with exact mathematical adherence.
3. **Global Acoustic & Pressure Wave Transport**:
   Pressure pulses propagate across the entire $H \times W$ grid in a single layer via $K$ landmark transport slots, avoiding the need for 50+ convolutional layers.

---

## 4. Product Features & Architecture
- **Multi-Scale Mesh Hierarchy**: Natively runs on regular or quadtree grid hierarchies.
- **Physics-Informed Loss Integration**: Conservation of mass, momentum, and energy directly verified in the residual combine stage.
- **Real-Time Interactive Design**: 1000 FPS simulation speeds on standard workstation GPUs (e.g., RTX 5060 Ti).

---

## 5. Target Market & Monetization
- **Primary Market**: Automotive aerodynamics, aerospace simulation, electronics cooling, clean energy (fusion & wind turbine modeling).
- **Business Model**: SaaS simulation API billed per compute-hour; on-premise CAD plugin licenses (Ansys, COMSOL, Siemens integration).
