#include "Wm5Vector3.h"
#include "Wm5ContSymBox3.h"
#include "Wm5Memory.h"
#include "Wm5Query.h"
#include "mex.h"

using namespace Wm5;

void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]){

    const mxArray *mVertices;
    const mxArray *mSamples;
    const mxArray *mFaces;
    const mxArray *mOrigFaces;
    
    double *vertices, *samples, *faces, *orig_faces;
    double *out_box, *out_planes;
    int nV, nS, nF, nOF;
    
    /* Check number of input and output parameters */
    if (nrhs != 4){
        mexErrMsgTxt("Must have 4 input arguments");
    }
    if (nlhs != 2){
        mexErrMsgTxt("Must have 2 output arguments");
    }

    /* Get matlab inputs */
    mSamples = prhs[0];
    mVertices = prhs[1];
    mFaces = prhs[2];
    mOrigFaces = prhs[3];
    
    /* Get dimensions of input data */
    nV = mxGetM(mVertices);
    nS = mxGetM(mSamples);
    nF = mxGetM(mFaces);
    nOF = mxGetM(mOrigFaces);
    
    /* Get pointer to input data */
    vertices = mxGetPr(mVertices);
    samples = mxGetPr(mSamples);
    faces = mxGetPr(mFaces);
    orig_faces = mxGetPr(mOrigFaces);
    

    /* Compute box */
    Vector3<double> *points, *samps, *facs, *orfacs;
    //SymBox3<double> box;

    /* Transfer points from one structure to the other */
    points = new1<Vector3d>(nV);
    for (int i = 0; i < nV; i++){
        points[i] = Vector3<double>(vertices[i], vertices[nV+i], vertices[2*nV+i]);
    }
    
    samps = new1<Vector3d>(nS);
    for (int i = 0; i < nS; i++){
        samps[i] = Vector3<double>(samples[i], samples[nS+i], samples[2*nS+i]);
    }
    
    facs = new1<Vector3d>(nF);
    for (int i = 0; i < nF; i++){
        facs[i] = Vector3<double>(faces[i]-1, faces[nF+i]-1, faces[2*nF+i]-1);
    }
    
    orfacs = new1<Vector3d>(nOF);
    for (int i = 0; i < nOF; i++){
        orfacs[i] = Vector3<double>(orig_faces[i]-1, orig_faces[nOF+i]-1, orig_faces[2*nOF+i]-1);
    }

    /* Compute box */
    double epsilon = 0.00001;
    Query::Type queryType = Query::QT_REAL;
	//Query::Type queryType = Query::QT_RATIONAL;
    SymBox3<double> bx = SymBox3<double>(nS, nV, nF, nOF, samps, points, facs, orfacs, epsilon, queryType);
    delete1(points);
    delete1(samps);
    delete1(facs);
    delete1(orfacs);
    
    int prob = bx.getProblem();
    if(prob)
    {
        plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
        plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
        return;
    }
    
    /* Create output data */
    plhs[0] = mxCreateDoubleMatrix(3, 5, mxREAL);
    
    /* Get pointer to output data */
    out_box = mxGetPr(plhs[0]);
    
    Box3<double> box = (Box3<double>) bx;
    
    /* Transfer box to output */
    out_box[0] = box.Center[0];
    out_box[1] = box.Center[1];
    out_box[2] = box.Center[2];
    out_box[3] = box.Axis[0][0];
    out_box[4] = box.Axis[0][1];
    out_box[5] = box.Axis[0][2];
    out_box[6] = box.Axis[1][0];
    out_box[7] = box.Axis[1][1];
    out_box[8] = box.Axis[1][2];
    out_box[9] = box.Axis[2][0];
    out_box[10] = box.Axis[2][1];
    out_box[11] = box.Axis[2][2];
    out_box[12] = box.Extent[0];
    out_box[13] = box.Extent[1];
    out_box[14] = box.Extent[2];
    
    int numBoxes = bx.getNumBoxes();
    
    if(numBoxes == 0)
    {
        plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
        return;
    }
    
    plhs[1] = mxCreateDoubleMatrix(18, numBoxes, mxREAL);
    out_planes = mxGetPr(plhs[1]);
    
    //double* normals = bx.getNormals();
    //double* ppoints = bx.getPoints();
    double* extents = bx.getExtents();
    double* axes = bx.getAxes();
    double* centers = bx.getCenters();
    double* scores = bx.getScores();
    
    for(int i = 0; i < numBoxes; i++)
    {
        
        out_planes[18*i] = centers[3*i];
        out_planes[18*i+1] = centers[3*i+1];
        out_planes[18*i+2] = centers[3*i+2];
        
        out_planes[18*i+3] = axes[9*i];
        out_planes[18*i+4] = axes[9*i+1];
        out_planes[18*i+5] = axes[9*i+2];
        out_planes[18*i+6] = axes[9*i+3];
        out_planes[18*i+7] = axes[9*i+4];
        out_planes[18*i+8] = axes[9*i+5];
        out_planes[18*i+9] = axes[9*i+6];
        out_planes[18*i+10] = axes[9*i+7];
        out_planes[18*i+11] = axes[9*i+8];
        
        out_planes[18*i+12] = extents[3*i];
        out_planes[18*i+13] = extents[3*i+1];
        out_planes[18*i+14] = extents[3*i+2];
        
        out_planes[18*i+15] = scores[3*i];
        out_planes[18*i+16] = scores[3*i+1];
        out_planes[18*i+17] = scores[3*i+2];
        
        /*
        out_planes[7*i] = normals[3*i];
        out_planes[7*i+1] = normals[3*i+1];
        out_planes[7*i+2] = normals[3*i+2];
        out_planes[7*i+3] = ppoints[3*i];
        out_planes[7*i+4] = ppoints[3*i+1];
        out_planes[7*i+5] = ppoints[3*i+2];
        out_planes[7*i+6] = scores[i];
        */
    }
    
	/*
	int* hullVerts = bx.getHullVerts();
	plhs[2] = mxCreateDoubleMatrix(1, 3*numBoxes, mxREAL);
	double* temp = mxGetPr(plhs[2]);

	for(int bla = 0; bla < 3*numBoxes; bla++)
	{
		temp[bla] = hullVerts[bla];
	}
	*/

    bx.freeFields();
    
}
