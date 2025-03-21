#include <ai.h>

extern const AtNodeMethods* ClassicKuwaharaImagerMtd;
extern const AtNodeMethods* AnisotropicKuwaharaImagerMtd;

enum
{
    CLASSIC_KUWAHARA_IMAGER,
    ANISOTROPIC_KUWAHARA_IMAGER
};

node_loader
{
    switch(i)
	{
    case CLASSIC_KUWAHARA_IMAGER:
        node->methods     = (AtNodeMethods*) ClassicKuwaharaImagerMtd;
        node->output_type = AI_TYPE_NODE;
        node->name        = "imager_ooClassicKuwahara";
        node->node_type   = AI_NODE_IMAGER;
        break;

    case ANISOTROPIC_KUWAHARA_IMAGER:
        node->methods     = (AtNodeMethods*) AnisotropicKuwaharaImagerMtd;
        node->output_type = AI_TYPE_NODE;
        node->name        = "imager_ooAnisotropicKuwahara";
        node->node_type   = AI_NODE_IMAGER;
        break;

    default:
        return false;
    }

   strcpy(node->version, AI_VERSION);
   return true;
}
