//#include "Wm5Vector3.h"
//#include "Wm5ContMinBox3.h"
//#include "Wm5Memory.h"
//#include "Wm5Query.h"
//#include "mex.h"

#include "Wm5Vector3.h"
//#include "Wm5Line3.h"
#include "Wm5Triangle3.h"
//#include "Wm5IntrLine3Triangle3.h"
#include "Wm5DistPoint3Triangle3.h"
#include "Wm5Memory.h"
//#include "Wm5Query.h"
#include "mex.h"

using namespace Wm5;

void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]){

    const mxArray *mVertices;
    const mxArray *mSamples;
    const mxArray *mPlane;
    const mxArray *mFaces;
    
    double *vertices, *samples, *plane, *faces;
    double *out;
    int nS, nP, nV, nF;

    /* Check number of input and output parameters */
    if (nrhs != 4){
        mexErrMsgTxt("Must have 4 input arguments");
    }
    if (nlhs != 1){
        mexErrMsgTxt("Must have 1 output arguments");
    }

    /* Get matlab inputs */
    mSamples = prhs[0];
    mPlane = prhs[1];
    mVertices = prhs[2];
    mFaces = prhs[3];

    /* Get dimensions of input data */
    nS = mxGetM(mSamples);
    nP = mxGetM(mPlane);
    nV = mxGetM(mVertices);
    nF = mxGetM(mFaces);

    /* Get pointer to input data */
    samples = mxGetPr(mSamples);
    plane = mxGetPr(mPlane);
    vertices = mxGetPr(mVertices);
    faces = mxGetPr(mFaces);

    /* Create output data */
    plhs[0] = mxCreateDoubleMatrix(nS, 2, mxREAL);

    /* Get pointer to output data */
    out = mxGetPr(plhs[0]);

    
    Vector3<double> *samps;
    Vector3<double> *verts;
    Vector3<double> *facs;
    //Vector3<double> *pl;
    
    /* Transfer points from one structure to the other */
    
    samps = new1<Vector3d>(nS);
    for (int i = 0; i < nS; i++){
        samps[i] = Vector3<double>(samples[i], samples[nS+i], samples[2*nS+i]);
    }
    
    verts = new1<Vector3d>(nV);
    for (int i = 0; i < nV; i++){
        verts[i] = Vector3<double>(vertices[i], vertices[nV+i], vertices[2*nV+i]);
    }
    
    facs = new1<Vector3d>(nF);
    for (int i = 0; i < nF; i++){
        facs[i] = Vector3<double>(faces[i], faces[nF+i], faces[2*nF+i]);
    }
    
    double a = plane[0];
    double b = plane[1];
    double c = plane[2];
    double d = plane[3];
    
    
    
    for(int i = 0; i < nS; i++) {
        // reflect the sample point across the plane
        double temp = 2 * (a * samps[i].X() + b * samps[i].Y() + c * samps[i].Z() + d);
		temp = temp / (pow(a,2)+pow(b,2)+pow(c,2));
        double x1 = samps[i].X() - temp * a;
        double y1 = samps[i].Y() - temp * b;
        double z1 = samps[i].Z() - temp * c;
        
        Vector3<double> p(x1, y1, z1);

        double pdist = a*x1 + b*y1 + c*z1 + d;
        
        double minDist = 1000;
        int minFace = -1;
        
        for(int j = 0; j < nF; j++) {
            int v1 = facs[j].X();
            int v2 = facs[j].Y();
            int v3 = facs[j].Z();
            
            Triangle3<double> tri(verts[v1-1], verts[v2-1], verts[v3-1]);
            
            DistPoint3Triangle3<double> dist(p, tri);
            //double distance = dist.Get();
            //For some reason dist.Get() was getting an error of
            //Sqrt(negative number). So, GetSquared allows to check that
            //if tht case is happening
            double distance = dist.GetSquared();
            if (distance < 0.0){
                distance = 0.0;
            }
            if(distance < minDist) {
                minDist = distance;
                minFace = j + 1;
            }
                
        }
        
        //out[i] = minDist / pdist; // weight by the distance to the plane
        out[i] = minDist;
        out[nS+i] = minFace;
        
    }
    
    //out[0] = minDist;
    //out[1] = minFace;

    delete1(samps);
    delete1(verts);
    delete1(facs);
}
