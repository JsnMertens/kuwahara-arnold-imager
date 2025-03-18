#include <ai.h>

AI_IMAGER_NODE_EXPORT_METHODS(ImagerMtd);

node_parameters
{
   AiParameterRGBA("color", 0.f, 1.f, 0.f, 1.f);
}

node_initialize
{
}

node_update
{
}

namespace
{
   static AtString rgba_str("RGBA");
   static AtString color_str("color");
}

imager_evaluate
{
   int aov_type = 0;
   const void *bucket_data;
   AtString output_name;
   AtRGBA color = AiNodeGetRGBA(node, color_str);
   while (AiOutputIteratorGetNext(iterator, &output_name, &aov_type, &bucket_data))
   {
      if (output_name != rgba_str)
         continue;
      AtRGBA* rgba = (AtRGBA*)bucket_data;
      for (int y = 0; y < bucket_size_y; ++y)
      for (int x = 0; x < bucket_size_x; ++x)
      {
         int idx = y * bucket_size_x + x;
         rgba[idx] = color;
      }
   }
}

node_finish
{
}

node_loader
{
   if (i>0) return false;
   node->methods     = (AtNodeMethods*) ImagerMtd;
   node->output_type = AI_TYPE_NODE;
   node->name        = "simple_imager";
   node->node_type   = AI_NODE_IMAGER;
   strcpy(node->version, AI_VERSION);
   return true;
}
