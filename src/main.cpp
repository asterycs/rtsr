#include "igl/readOFF.h"
#include "igl/viewer.h"
#include <iostream>
#include <set>
#include <stack>
#include <limits>
#include <list>

bool callback_key_down(Viewer &viewer, unsigned char key, int modifiers) {
    std::cout << "Keyboard callback!" << std::endl;

    return true;
}

bool callback_load_mesh(Viewer& viewer,string filename)
{
  igl::readOFF(filename,P,F,N);
  callback_key_down(viewer,'1',0);
  return true;
}

int main(int argc, char *argv[]) {
    // Read points and normals
    // igl::readOFF(argv[1],P,F,N);

    Viewer viewer;
    viewer.callback_key_down = callback_key_down;

    viewer.launch();
}
