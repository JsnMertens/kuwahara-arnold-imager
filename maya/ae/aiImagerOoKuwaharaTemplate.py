import maya.cmds as cmds
import maya.mel
from mtoa.ui.ae.shaderTemplate import ShaderAETemplate
from mtoa.ui.ae.aiImagersBaseTemplate import ImagerBaseUI, registerImagerTemplate


class AEaiImagerOoKuwaharaTemplate(ShaderAETemplate):
    def setup(self):
        self.beginScrollLayout()
        currentWidget = cmds.setParent(query=True)
        self.ui = ImagerOoKuwaharaUI(parent=currentWidget, nodeName=self.nodeName, template=self)

        maya.mel.eval("AEdependNodeTemplate {}".format(self.nodeName))

        self.addExtraControls()
        self.endScrollLayout()


class ImagerOoKuwaharaUI(ImagerBaseUI):
    def __init__(self, parent=None, nodeName=None, template=None):
        super(ImagerOoKuwaharaUI, self).__init__(parent, nodeName, template)

    def setup(self):
        super(ImagerOoKuwaharaUI, self).setup()
        self.beginLayout("Main", collapse=False)
        self.addControl("radius",
            label="Radius",
            annotation="Size of the Kuwahara filter.",
            hideMapButton=True
        )
        self.endLayout()


registerImagerTemplate("aiImagerOoKuwahara", ImagerOoKuwaharaUI)
