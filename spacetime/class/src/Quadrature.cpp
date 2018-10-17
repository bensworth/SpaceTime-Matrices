#include "Quadrature.hpp"
#include <cmath>
#include <sstream>
#include <string>
#include <iostream>
#include <fstream>
#include <cstring>
#include <stdlib.h>
#include <stdio.h>

Quadrature::Quadrature(const char *quadType, const int order)
    : order(order), numAngles(order * (order + 2)), mu(numAngles, 0.0),
      eta(numAngles, 0.0), xi(numAngles, 0.0), w(numAngles, 0.0) {

  std::ifstream quadFile;
  quadFile.open(quadType);
  std::string line("");
  int count = 0;
  std::string muStr, etaStr, xiStr, wStr;
  if (quadFile.is_open()) {
    for (int i = 0; i < numAngles + 1; i++) {
      std::getline(quadFile, line);
      if (line.at(0) != 'L' && line.at(0) != 'Q') {
        std::istringstream iss(line);
        iss >> muStr >> etaStr >> xiStr >> wStr;
        mu[count] = atof(muStr.c_str());
        eta[count] = atof(etaStr.c_str());
        xi[count] = atof(xiStr.c_str());
        w[count] = atof(wStr.c_str());
      } else {
        count--;
      }
      count++;
    }
#if 0
    // Normalize weights to 4*pi
  double sumWeight = 0;
  for (int i = 0; i < numAngles; i++) {
    sumWeight += w[i];
  }
  for (int i = 0; i < numAngles; i++) {
    w[i] = w[i] * 4.0 * M_PI / sumWeight;
    //std::cout << mu[i] << "  " << eta[i] << "  " << w[i] << std::endl;
  }
#endif
    quadFile.close();
  } else {
    std::cout << "Unable to open file: " << quadType << std::endl;
  }
}


Quadrature::Quadrature(const int order)
    : order(order), numAngles(order * (order + 2)), mu(numAngles, 0.0),
      eta(numAngles, 0.0), xi(numAngles, 0.0), w(numAngles, 0.0) 
{
	
	
  std::ifstream quadFile;
  if (order == 2)
  {
  	  quadFile.open("./SN_quad_angles/LS_1.txt");
  }
  else if (order == 4)
  {
  	  quadFile.open("./SN_quad_angles/LS_2.txt");
  }
  else if (order == 8)
  {
  	  quadFile.open("./SN_quad_angles/LS_3.txt");
  }
  else
  {
  	  std::cout << "Unable to use this quadrature order " << std::endl;
  }
  
  std::string line("");
  int count = 0;
  std::string muStr, etaStr, xiStr, wStr;
  if (quadFile.is_open()) {
    for (int i = 0; i < numAngles + 1; i++) {
      std::getline(quadFile, line);
      if (line.at(0) != 'L' && line.at(0) != 'Q') {
        std::istringstream iss(line);
        iss >> muStr >> etaStr >> xiStr >> wStr;
        mu[count] = atof(muStr.c_str());
        eta[count] = atof(etaStr.c_str());
        xi[count] = atof(xiStr.c_str());
        w[count] = atof(wStr.c_str());
      } else {
        count--;
      }
      count++;
    }
#if 0
    // Normalize weights to 4*pi
  double sumWeight = 0;
  for (int i = 0; i < numAngles; i++) {
    sumWeight += w[i];
  }
  for (int i = 0; i < numAngles; i++) {
    w[i] = w[i] * 4.0 * M_PI / sumWeight;
    //std::cout << mu[i] << "  " << eta[i] << "  " << w[i] << std::endl;
  }
#endif
    quadFile.close();
  }
  else {
    std::cout << "Unable to open file: " << std::endl;
  }
    	
}



std::vector<double> Quadrature::getOmega(const int angle) const
{
    if (is_2D_)
    {
        std::vector<double> omega(2);
        omega[0] = mu[angle];
        omega[1] = eta[angle];
        return omega;
    }
    else
    {
        std::vector<double> omega(3);
        omega[0] = mu[angle];
        omega[1] = eta[angle];
        omega[2] = xi[angle];
        return omega;
    }
}







