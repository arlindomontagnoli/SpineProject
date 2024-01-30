<h2>DrSpine and SpineSegmentation</h2>
<p>1 - Execute 'DrSpine.py' script to initiate the process, and load your X-Ray image, whether it's 'Model.jpeg' or your own. The X-ray Model comes pre-landmarked to simplify the software test.</p>

<p>2 -To execute 'DrSpine.py', landmarking is required. Run the 'SpineSegmentation.py' script to landmark your X-ray image.
JPEG, PNG and DICOM extension are supported.</p>

<p>After opening the Xray, select a Region of Interest (ROI) simply by clicking and dragging the mouse cursor on the image. A new window will pop up with the selected region. Use the left mouse button to add landmarks and the right mouse button to remove any incorrect ones. It's important to note that the order in which you landmark the vertebrae is not relevant.</p>

<p>Ensure to landmark a set of four points for each vertebra, spanning from the first sacral vertebra (S1) to the second cervical vertebra (C2), including four points for the femoral head.</p>

<p>To keep the landmarks annotated within the ROI, press the <enter> key. The landmarks will be inserted in the main Xray window.</p>

<p>The landmarks will be automatically saved when exiting the main window, by pressing the <esc> key.</p>


<h3>Python packages dependencies list:</h3>
<p>pydicom, numpy, pillow, matplotlib, gdcm, scipy, pandas, opencv, screeninfo, pysimplegui, scikit-learn</p>

