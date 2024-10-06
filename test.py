# https://github.com/TransformerLensOrg/TransformerLens

import transformer_lens

# Load a model (eg GPT-2 Small)
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

print(model)

# Run the model and get logits and activations
logits, activations = model.run_with_cache("Hello World")

print(logits)
print("--")
print(activations)
