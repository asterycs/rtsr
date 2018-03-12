#include "igl/readOFF.h"
#include "igl/viewer/Viewer.h"
#include <iostream>
#include <set>
#include <stack>
#include <limits>
#include <list>

bool callback_key_down(igl::viewer::Viewer &viewer, unsigned char key, int modifiers) {
    std::cout << "Keyboard callback!" << std::endl;

    return true;
}


int main(int argc, char *argv[]) {
    // Read points and normals
    // igl::readOFF(argv[1],P,F,N);

    igl::viewer::Viewer viewer;
    viewer.callback_key_down = callback_key_down;

    viewer.launch();
}
