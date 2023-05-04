# Adding constrain 
###to add a constrain, four places need to be edited

1. add a function in reader_zoo.py to read data, and add an annotation above it<br>
name in the annotation will be the name of the constrain<br>
example as follows<br>
#####params of the function
   :prefix： function will load the image data according prefix and suffix.<br>
   :returns: a dict include at least a k,v pair which data should be: "img":{data of image}   

```python
@ReaderZoo.register("simi")
def read_simi_img(prefix):
    simi_img = read_tiff_stack(prefix + "_process.tiff")
    simi_img = _gauss(simi_img)
    simi_img = simi_img.astype(np.float32)
    simi_img = simi_img[np.newaxis, ...]
    simi_img = (simi_img - np.min(simi_img)) / (np.max(simi_img) - np.min(simi_img))
    simi = {"img": simi_img}
    return simi
```

2. add a function in deformer_zoo.py to perform deform space to the new constrain.
annotation is also used in this part.<br>
Or new constrain can just add the former annotation.
   
   #####params:
      
      fix：fix data dict, shall be exactly the same as defined in first step
            
      mov: mov data dict
      
      deform_space: deform space
   
   #####returns:
      
      a dict, at least contains a key img


```python
@DeformerZoo.register("simi",
                      "outline",
                      "convex",
                      "hpf",
                      "hole",
                      )
def deform_img(fix: dict, mov: dict, deform_space: torch.Tensor) -> dict:
    reg = {"img": grid_sample(mov["img"], deform_space)}
    return reg
```

3.loss. Define a loss function and add the annotation. Forward function should contains four params  
*remember tot import the loss function in the __init__.py file*<br>
example as follows<br>
```python
@LossZoo.register("outline", "hpf", "convex")
class RegMSE(nn.Module):
    def __init__(self):
        super(RegMSE, self).__init__()
        pass

    def forward(self, fix: dict, mov: dict, reg: dict, deform_space: torch.Tensor):
        return mse_loss(reg["img"], fix["img"])
```

4. add a function in metric_zoo.py to evaluate the registration, and add an annotation. If not added, the new
constrain won't be evaluate.<br>
example as follows<br>
   ```python
   @MetricZoo.register("outline", "convex", "hpf")
   def iou_metric(fix: dict, reg: dict):
       """
       :param fix: fix image data
       :param reg: registration image data
       :return:
       """
       fix_img, reg_img = fix["img"].detach().cpu().numpy() * 255, reg["img"].detach().cpu().numpy() * 255
       return cal_iou(fix_img, reg_img)
   ```

5. remember to add the new constrain in config file(yaml file).<br>
   Including three part, TrainConfig for data name; ModelConfig for constrain;
   and LossConfig for loss function
   

## 附录
current constrain are as follows:

1. vc  include VL(lateral ventricle), V3(third ventricle),V4(fourth ventricle).
2. HPF Hippocampal formation, except some regions including Entorhinal area.
3. CP Caudoputamen.
4. CB ("Cerebellar cortex", 11),
        ("arbor vitae", 11)
5. BS Brain stem
6. CTX Cerebral cortex.  Hippocampal formation is removed
7. CBX Cerebellar cortex