# Flow JEPA

Experiments in lossless representation learning.

Paper is a WIP, taking a more experimentalist tone because it feels more honest, citations are shit, lots of experimental stuff to add, a bit to vibey at the moment mainly because I don't like writing tex and models like improvising a lil on what I wrote, but i guess one of the nice things about being an outsider is you don't need to follow any process. I'll iterate on this in public.

Basically trying to see if we can decouple learning semantics from destroying information. Most perception models toss out "noise" to get invariance. This architecture tries to keep everything (via a bijective flow) but shove the noise into a specific sub-manifold, whilst also trying to formulate it so it has desirable properties for representational space planning.

This should mostly be ready for scale up to a reasonable point, got some polishing to do to make big boi ready but it supports jagged / dynamic sized inputs, flattened block sparse etc etc. Lots of parts to ablate etc.

## The Gist

Backbone: Stochastic Masked Normalizing Flow (preserves exact density, makes the flow task more perceptual).

Prior: Conditional VAE with a Linear Encoder (forces the flow to straighten the manifold).

Objective: Joint Embedding Predictive Architecture (JEPA) style, lejepa but lossless.

Hardware: Trained on a TinyBox (6x 4090).

Results: ~87.6 - 90.5% linear probe on Imagenette whilst being fully invertable/perfect reconstruction. (ABLATE ME I NEED GPUS)

Status: Kinda works is the best kind of works. 

Citation

If you use this, or just think it's neat:
```
@misc{hibble2025flowjepa,
  title={Flow Jepa: Experiments in Bijective Isometry},
  author={Adam Hibble},
  year={2025},
  note={Draft v0.0.1}
}
```

Contact

Adam Hibble (@algomancer)
