import maya.cmds as cmds
import maya.mel
from mtoa.ui.ae.shaderTemplate import ShaderAETemplate
from mtoa.ui.ae.aiImagersBaseTemplate import ImagerBaseUI, registerImagerTemplate


class AEaiImagerOoAnisotropicKuwaharaTemplate(ShaderAETemplate):
    def setup(self):
        self.beginScrollLayout()
        currentWidget = cmds.setParent(query=True)
        self.ui = ImagerAnisotropicKuwaharaUI(parent=currentWidget, nodeName=self.nodeName, template=self)

        maya.mel.eval("AEdependNodeTemplate {}".format(self.nodeName))

        self.addExtraControls()
        self.endScrollLayout()


class ImagerAnisotropicKuwaharaUI(ImagerBaseUI):
    def __init__(self, parent=None, nodeName=None, template=None):
        super(ImagerAnisotropicKuwaharaUI, self).__init__(parent, nodeName, template)

    def setup(self):
        super(ImagerAnisotropicKuwaharaUI, self).setup()
        self.beginLayout("Main", collapse=False)
        self.addControl(
            "radius",
            label="Radius",
            annotation="Size of the Kuwahara filter.",
            hideMapButton=True
        )
        self.addControl(
            "eccentricity",
            label="Eccentricity",
            annotation="Ellipse eccentricity, zero for isotropic.",
            hideMapButton=True
        )
        self.addControl(
            "sharpness",
            label="Sharpness",
            annotation="Sharpness filter.",
            hideMapButton=True
        )
        self.addControl(
            "tensor_size",
            label="Tensor Size",
            annotation="Structure Tensor Gaussian Size.",
            hideMapButton=True
        )
        self.endLayout()


registerImagerTemplate("aiImagerOoAnisotropicKuwahara", ImagerAnisotropicKuwaharaUI)
