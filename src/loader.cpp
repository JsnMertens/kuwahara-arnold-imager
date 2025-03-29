#include <ai.h>

extern const AtNodeMethods* KuwaharaImagerMtd;
extern const AtNodeMethods* AnisotropicKuwaharaImagerMtd;

enum
{
    KUWAHARA_IMAGER,
    ANISOTROPIC_KUWAHARA_IMAGER
};

node_loader
{
    switch(i)
	{
    case KUWAHARA_IMAGER:
        node->methods     = (AtNodeMethods*) KuwaharaImagerMtd;
        node->output_type = AI_TYPE_NODE;
        node->name        = "imager_ooKuwahara";
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
