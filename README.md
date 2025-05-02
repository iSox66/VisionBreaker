# VisionBreaker
Simple Adversarial Attack , Wont work agaist Neural Network Level

you might need python and git

https://www.python.org/downloads/
https://git-scm.com/downloads

right click and click "edit in Notepad" so you will be able to see the functions to set as "True" or "false" as needed

and below there are the intensities of it for those who cares and want to test it out 



YES you will need to name the picture "input.jpg"



Maybe FGSM, PGD, Carlini-Wagner(CW) coming soon tooo if I dont forget (since I have other things to focus on ... this is just what I wanted to do with my spare time)




JPEG / JPG super common, compressed, lossy
PNG lossless, supports transparency
BMP uncompressed, giant files, but works fine
TIFF / TIF used in pro/medical imaging, supported yes
GIF but only first frame (Pillow can read it, but OpenCV treats it like a static image)
WEBP yep.. works if your Pillow has WEBP support (modern versions usually do)
PPM / PGM / PBM ... niche formats, but supported



RAW camera formats like .CR2, .NEF, .ARW... nope, youd need a raw converter first
HEIF / HEIC (used by iPhones) only if you have Pillow compiled with HEIF plugin (usually no by default)
