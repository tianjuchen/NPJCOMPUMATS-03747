# DeepONet X\_loc Spatial Continuous Method

The analogy of `X_loc` as "pixel coordinates in a video frame" captures a critical role of the trunk network in DeepONet: it grounds the model’s predictions in **explicit spatial positions**, enabling the network to learn how features (e.g., microstructure properties, pixel intensities) vary across a spatial domain. Let’s unpack this in more detail, focusing on how `X_loc` empowers the trunk network to model spatial variation:

### 1. `X_loc` as "Spatial Addresses": Defining "Where" to Predict

In a video frame, each pixel has a unique coordinate (e.g., `(x, y)` in 2D), which acts as a "spatial address"—a precise label for where that pixel is located within the frame. Similarly, `X_loc` in the preprocessing function serves as a normalized "spatial address" for the microstructure domain.



* In the code, `X_loc` is constructed as `np.array(index) / img_size`, which normalizes spatial indices to the range `[0, 1]`. This is analogous to scaling pixel coordinates in a video frame (e.g., from `(0, 0)` to `(width-1, height-1)`) to a standardized range like `[0, 1]` for consistent processing.

* Each entry in `X_loc` corresponds to a specific point in the spatial domain of the microstructure (e.g., a position in a 1D, 2D, or 3D grid where properties like grain size or phase fraction are measured).

### 2. The Trunk Network: Learning "How Features Vary with Spatial Address"

The trunk network in DeepONet is explicitly designed to process `X_loc` (spatial addresses) and learn **the relationship between spatial position and feature values**. In the video analogy:



* If the video frame contains a gradient (e.g., dark on the left, light on the right), the trunk network would learn that pixel value increases as `x` (horizontal coordinate) increases.

* For a microstructure, this could mean learning that "grain boundaries are more dense near position `0.2`" or "phase transformation starts earlier at position `0.8`"—patterns that depend on spatial location.

The trunk network typically uses layers like fully connected networks (FCNs) with smooth activation functions (e.g., sine, ReLU) to model these spatial relationships. Unlike convolutional layers (which exploit local spatial correlations), the trunk network treats `X_loc` as explicit coordinates, allowing it to learn **global spatial patterns** (e.g., how features vary across the entire domain, not just locally).

### 3. Why This Design Supports Spatial Extrapolation

A key advantage of encoding spatial positions via `X_loc` is that it enables the trunk network to **extrapolate to unseen spatial positions**—critical for microstructure evolution, where predicting behavior at unmeasured locations is often necessary.



* In the video example: If the trunk network learns that pixel intensity increases linearly with `x` in the training range `x ∈ [0, 0.6]`, it can extrapolate to `x = 0.8` by extending that linear pattern.

* For microstructures: If `X_loc` during training covers positions `[0, 0.5]`, and the trunk network learns that "grain size increases with position" in this range, it can predict grain size at `x = 0.7` by continuing that trend—even if `x = 0.7` was never seen in training.

This is far more robust than relying on implicit spatial patterns (e.g., from convolutional layers), which struggle with extrapolation beyond the training spatial range.

### 4. Coupling with `X_func`: "What" + "Where" = Predictions

The trunk network’s spatial knowledge (from `X_loc`) is useless in isolation—it must be paired with the branch network’s understanding of "what" features exist in the microstructure (from `X_func`).



* In the video analogy: `X_func` would represent the content of the frame (e.g., edges, textures), while `X_loc` tells the model "which pixel to focus on." The trunk network’s spatial patterns (e.g., "intensity increases with `x`") are modulated by the branch network’s content (e.g., "but only in regions with a vertical edge").

* For microstructures: `X_func` encodes the "content" (e.g., initial grain distribution, alloy composition), and `X_loc` specifies "where to predict." The trunk network’s learned spatial trends (e.g., "phase A dominates at `x < 0.3`") are combined with `X_func`’s features (e.g., "if initial phase B is high, phase A spreads slower") to produce location-specific predictions.

### 5. Why Normalization of `X_loc` Matters

The normalization of `X_loc` to `[0, 1]` (via `index / img_size`) is critical for this framework. It ensures that:



* Spatial positions are invariant to the absolute size of the domain (e.g., a microstructure of size 128 vs. 256 is treated consistently).

* The trunk network can learn generalizable spatial patterns (e.g., "near the boundary" vs. "in the center") rather than memorizing arbitrary index values.

### Summary

Treating `X_loc` as "pixel coordinates in a video frame" highlights its role as a **spatial grounding mechanism**. The trunk network uses these coordinates to learn how microstructure features vary across space—capturing trends like gradients, boundaries, or regional heterogeneities. By separating spatial position encoding (`X_loc`) from feature content encoding (`X_func`), DeepONet gains the flexibility to extrapolate to new spatial locations, making it uniquely suited for predicting microstructure evolution across unmeasured regions.



# DeepONet and Microstructural Extrapolation

In DeepONet frameworks designed to extrapolate microstructure evolution (a time-dependent process), even if `X_loc` itself is not explicitly a time-dependent function, the model can still capture temporal dynamics through **implicit encoding of time within the "function input" (**`X_func`**)** and **learning the coupling between spatial features and temporal evolution patterns**. Here’s how this works within the framework:

### 1. **Time is Encoded in the "Function Input" (**`X_func`**), Not&#x20;**`X_loc`

DeepONet’s core is to learn a mapping from a "source function" (via the branch network) to a "target function" (via the trunk network). For microstructure evolution (a spatiotemporal process), **time is typically embedded in the source function&#x20;**`X_func`, while `X_loc` focuses on spatial coordinates.



* In the provided `preprocess` function, `X_func` is structured as a spatial grid (`14×14`) with multiple channels (adjusted to `target_channels=12`). These channels or the samples within `X_func` can implicitly represent:


  * **Snapshots of microstructure at different times**: For example, each channel in `X_func` could correspond to a microstructure state at a specific time step (e.g., `t=0, t=1, ..., t=11` for 12 channels).

  * **Temporal features derived from evolution**: `X_func` might encode time-dependent properties (e.g., grain growth rate, phase transformation kinetics) as spatial features, which the branch network learns to associate with temporal progression.

* Thus, even though `X_loc` (spatial coordinates) is not time-dependent, the branch network processes `X_func` to extract **temporal patterns** (e.g., "how microstructures evolve from `t=0` to `t=10`") alongside spatial features.

### 2. **The Trunk Network (**`X_loc`**) Handles Spatial Extrapolation, While Temporal Extrapolation Relies on the Branch Network**

Microstructure evolution requires extrapolation in **both space and time**. DeepONet splits this responsibility:



* **Spatial extrapolation**: `X_loc` (normalized spatial indices) allows the trunk network to learn how microstructure properties vary across spatial locations (e.g., "grain size at position `x=0.3` vs. `x=0.7`"). Even for unseen spatial positions, the trunk network generalizes using the spatial patterns learned from `X_loc`.

* **Temporal extrapolation**: This is driven by the branch network’s understanding of `X_func`’s temporal embeddedness. For example:


  * If `X_func` contains time-series snapshots, the branch network learns a "temporal signature" (e.g., "grain boundaries coarsen linearly over time").

  * When extrapolating to a new time `t_new`, the model uses this signature to predict how the spatial features (processed by the trunk network) should evolve at that time.

### 3. **Coupling Spatial and Temporal Information via Feature Fusion**

DeepONet’s key innovation is the **fusion of branch and trunk network outputs**, which enables the model to link temporal patterns (from `X_func`) with spatial locations (from `X_loc`).



* The branch network compresses `X_func` into a latent vector that encodes both spatial features and their temporal dynamics (e.g., "at time `t`, the microstructure at spatial region `(i,j)` has a certain phase fraction").

* The trunk network processes `X_loc` to encode spatial coordinates into a latent vector (e.g., "position `x` is in a region prone to rapid phase transformation").

* These latent vectors are combined (via dot product or concatenation) to predict the microstructure state at the specific spatial location **for any time encoded in&#x20;**`X_func`—including times beyond the training range (extrapolation).

### 4. **Analogy to Spatiotemporal Modeling**

Think of microstructure evolution as a video:



* Each frame is a spatial snapshot at a specific time.

* `X_func` is like a stack of these frames (time embedded in channels/samples), allowing the branch network to learn "how frames change over time."

* `X_loc` is like a pixel coordinate in the video, allowing the trunk network to learn "how pixels vary across the frame."

* The model fuses these to predict "what pixel value will be at position `(x,y)` in a future frame (unseen time)."

### Summary

Even with a time-agnostic `X_loc`, DeepONet addresses temporal extrapolation of microstructure evolution by:



1. Embedding time into `X_func` (via time-series snapshots or temporal features).

2. Using the branch network to learn temporal patterns from `X_func`.

3. Fusing these temporal patterns with spatial information from `X_loc` to predict evolution at new times and locations.

This separation of responsibilities (branch for temporal/functional features, trunk for spatial locations) is what makes DeepONet effective for spatiotemporal extrapolation tasks.
