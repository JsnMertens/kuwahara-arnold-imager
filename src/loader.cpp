#include <ai.h>

extern const AtNodeMethods* KuwaharaImagerMtd;

enum
{
    KUWHARA_IMAGER = 0
};

node_loader
{
    switch(i)
	{
    case KUWHARA_IMAGER:
        node->methods     = (AtNodeMethods*) KuwaharaImagerMtd;
        node->output_type = AI_TYPE_NODE;
        node->name        = "ooKuwaharaImager";
        node->node_type   = AI_NODE_IMAGER;
        break;

    default:
        return false;
    }

   strcpy(node->version, AI_VERSION);
   return true;
}
