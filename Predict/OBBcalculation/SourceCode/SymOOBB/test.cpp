#include "Wm5Vector3.h"
#include "Wm5ContBox3.h"
#include "Wm5Memory.h"

using namespace Wm5;

int main(void){

    Vector3<double> *points;
    Box3<double> box;

    points = new1<Vector3d>(10);
    for (int i = 0; i < 10; i++){
        points[i] = Vector3<double>(0.0, 0.0, 0.0);
    }
    box = ContOrientedBox(10, points);
    delete1(points);

    return 0;
}
