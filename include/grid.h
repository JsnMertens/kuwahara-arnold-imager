#ifndef KUWAHARAARNOLD_GRID_H_
#define KUWAHARAARNOLD_GRID_H_

namespace grid
{
    struct GridPoint
    {
        int x;
        int y;

        constexpr GridPoint() : x(0), y(0) {}
        constexpr GridPoint(int x, int y) : x(x), y(y) {}
    };

    using GridSize = GridPoint;

    struct GridRegion
    {
        GridPoint min;
        GridPoint max;

        constexpr GridRegion() : min(), max() {}
        constexpr GridRegion(const GridPoint& min, const GridPoint& max) : min(min), max(max) {}
    };
    
} // namespace grid

#endif // KUWAHARAARNOLD_GRID_H_
