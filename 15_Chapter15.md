![Great prompt, dear reader! Here is a DALL-E image generation prompt for this chapter in less than 1000 characters:  `Generate an image of Dracula and Jonathan standing together with a PyTorch book, while Sundararajan Sellamanickam holds a crystal ball in the foreground. Dracula's castle should be visible in the background, and the weather should be sunny with a hint of rain clouds. Add a caption with the text "Quantizing Dynamic Input Shapes with Sundararajan Sellamanickam."`](https://oaidalleapiprodscus.blob.core.windows.net/private/org-ct6DYQ3FHyJcnH1h6OA3fR35/user-qvFBAhW3klZpvcEY1psIUyDK/img-awhQTdqZaGmrplxrCmVM4uvp.png?st=2023-04-13T23%3A53%3A03Z&se=2023-04-14T01%3A53%3A03Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-04-13T17%3A15%3A30Z&ske=2023-04-14T17%3A15%3A30Z&sks=b&skv=2021-08-06&sig=uG182GGLtYKIaZBWpyDx8uJUPce7QMKgz0Igsb1X2YA%3D)


# Chapter 15: Quantization with Dynamic Input Shapes

Welcome back, dear reader! In the last chapter, we learned how to convert a floating-point model to a quantized model. Continuing on this journey of Fine Tuning Quantized LLMs in PyTorch, we are excited to welcome a special guest on this chapter - Sundararajan Sellamanickam! Sundararajan is an expert in the field of deep learning and has published several papers on the topic of model quantization.

In this chapter, we will dive into the topic of quantization with dynamic input shapes. In real-world applications, we often encounter scenarios where input shapes vary during inference, making it challenging to apply quantization. However, quantization is essential to perform efficient inference on devices with limited resources such as mobile and IoT devices.

We will explore the various techniques that can be employed to quantize LLMs with dynamic input shapes. We will also investigate the impact of quantizing LLMs with dynamic input shapes on their accuracy.

Before we begin, let's have a word from our expert, Sundararajan.

> "Hello, readers! It's great to be joining you in this chapter. Quantizing models with dynamic input shapes is a fascinating topic, and I am excited to share my insights on this subject. As we go through this chapter, I encourage you to experiment with the code samples and see the impact of different techniques on model accuracy." 

Are you ready, dear reader? Let's dive in!
# Chapter 15: Quantization with Dynamic Input Shapes

Welcome back, dear reader! In the last chapter, we learned about converting a floating-point model to a quantized model. Continuing our journey of fine-tuning Quantized LLMs in PyTorch, we are excited to welcome a special guest in this chapter - Sundararajan Sellamanickam!

As we begin our chapter, let us take a trip down to Transylvania, a place shrouded in mysteries, myths, and legends. It was said that the infamous vampire, Count Dracula, lived in a small castle on the outskirts of a small town. Even though the castle had been abandoned for centuries, the people of the town still feared the legend of the vampire.

On one fateful day, the town witnessed a strange incident. They found that the output of their local weather forecasting model was not accurate. The model predicted rainfall, but the weather was clear and sunny. The townspeople were bewildered and approached the young prodigy, Jonathan, who excelled in computer programming, for help.

Jonathan knew that the inaccurate output was due to the model not being able to handle dynamic input shapes, which caused inaccuracies. As the only expert in town, Jonathan knew that he had to quantize the model. So, he sought out Sundararajan, a renowned expert in quantizing LLMs with dynamic input shapes, for his help.

Sundararajan helped Jonathan understand the difficulties in quantizing LLMs with dynamic input shapes and suggested several techniques that could be used to solve the problem. Sundararajan introduced Jonathan to the first technique - setting a fixed input shape to the quantization process, which helped Jonathan in successfully converting his model to a quantized model.

However, the weather forecasting model was not the only problem the town was facing. There were many other models, such as traffic management and waste management models, that were not able to handle dynamic input shapes. Jonathan thus put Sundararajan's teachings to use and began to fine-tune various LLMs with dynamic input shapes using different techniques suggested by Sundararajan.

After several attempts, Jonathan was finally able to achieve a high level of accuracy with the quantized models. The town was now able to make better predictions and decisions, thanks to the calibrated quantized models.

In conclusion, quantizing LLMs with dynamic input shapes is crucial to achieve efficient inference on devices with limited resources such as mobile and IoT devices. By applying the techniques taught in this chapter, we can produce accurate quantized models that can work efficiently even with dynamic input shapes.

We hope you enjoyed reading this chapter, and we want you to try the code samples to fine-tune quantized LLMs with dynamic input shapes on your own. We would also like to thank our special guest, Sundararajan, for his invaluable insights into the topic of model quantization, and we look forward to seeing you soon in the next chapter.
Certainly, dear reader! Here is an explanation of the code used to resolve the Dracula story:

To resolve the issue of inaccurate weather forecasting, our protagonist, Jonathan, used the techniques suggested by Sundararajan to quantize the forecasting model. The following code is an example of how to set a fixed input shape for the quantization process using PyTorch:

```python
import torch
import torch.quantization
import torch.nn as nn

# Load the pre-trained model.
model = torch.load('weather_forecasting_model.pt')

# Set the fixed input shape and run the model to calibrate the quantization parameters.
input_shape = (1, 3, 256, 256)
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model = torch.quantization.prepare(model, inplace=True)
_ = model(torch.randn(*input_shape))
model = torch.quantization.convert(model, inplace=True)
```

In the above code, we first load the pre-trained weather forecasting model. We then set the fixed input shape to (1, 3, 256, 256) using the `input_shape` variable. This sets the input shape to a fixed value for the quantization process.

Next, we set the model in evaluation mode using the `eval()` method and define a quantization configuration using `get_default_qconfig('fbgemm')`. We then prepare the model for quantization using `prepare()` method, which calculates the initial quantization parameters for each layer. 

We then run the model with dummy input tensors whose shape is equal to `input_shape` to determine optimal quantization parameters for the model. 

Finally, we convert the model into a quantized model using `convert()` and set `inplace=True` to modify the existing model object.

By applying the techniques of quantization with dynamic input shapes taught in this chapter, we can calibrate our models for efficient inference on devices with limited resources.


[Next Chapter](16_Chapter16.md)