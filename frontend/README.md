## Slogan

Generate/Remix your original song!

## Introduction

`Remixer` is a project supporting:

1. Music generation specified by genre
2. Apply music style(of a genre, or from any song) to an existing one -- Bang, you have your own remixed song!

## Model Architecture

The project is mostly inspired by the incredible image generation ability
of [VAE](https://en.wikipedia.org/wiki/Variational_autoencoder):

1. For the music generation feature: Simply sample a gaussian noise, and decode the noise with StableDiffusion. Once an
   audio image - representing the spectrogram of an audio - is generated, convert it into audio format(.wav)
2. For the `Remixer` part: Blending the ideas
   of [Neural Style Transfer](https://en.wikipedia.org/wiki/Neural_style_transfer) and `VAE`:
    * We have a trained encoder from the VAE. Apply it to "content" audio to get "content" latents. For the "style"
      part, we either use pretrained-genre embeddings, or you can upload your song.
    * Treat the twos embeddings as image, apply `NST` to get a styled song -- or a 'remixed' song
    * Convert it back to audio


## Flagging

If you are unsatisfied with the model output, you are welcomed to flag a report by clicking the "flagging" buttons
underneath.

We'll analyze the results and use them to improve the model!
