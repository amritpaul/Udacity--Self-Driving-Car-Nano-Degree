/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
std::default_random_engine randomEngine;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  
  if(is_initialized) return;
  
  num_particles = 100;

  double xStd = std[0];
  double yStd = std[1];
  double thetaStd = std[2];

  std::normal_distribution<double> initX(x,xStd);
  std::normal_distribution<double> initY(y,yStd);
  std::normal_distribution<double> initTheta(theta,thetaStd);

  for(int i=0; i< num_particles ; i++) {
      Particle particle_;
      particle_.id=i;
      particle_.x= initX(randomEngine);
      particle_.y=initY(randomEngine);
      particle_.theta = initTheta(randomEngine) ;
      particle_.weight = 1.0;
      particles.push_back(particle_);

  }
  is_initialized=true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  std::normal_distribution<double> X(0, std_pos[0]);
  std::normal_distribution<double> Y(0, std_pos[1]);
  std::normal_distribution<double> Theta(0, std_pos[2]);
  
  
  for(int i=0; i< num_particles; i++){
    if(fabs(yaw_rate) > 0.00001){
      particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
      particles[i].y += (velocity / yaw_rate) * (-cos(particles[i].theta + (yaw_rate * delta_t)) + cos(particles[i].theta));
      particles[i].theta += (yaw_rate * delta_t);
      
      particles[i].x += X(randomEngine);
      particles[i].y += Y(randomEngine);
      particles[i].theta += Theta(randomEngine);
    }
    else{
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  // Perform Nearest Neighbour Algorithm
  
  for(int i=0; i< observations.size(); i++){
    LandmarkObs landmarkObs_obs = observations[i];
    double minDistance = 100000000;
    int minIndex = -100000000;
    
    for(int j=0; j< predicted.size(); j++){
      LandmarkObs landmarkObs_pred = predicted[j];
      
      double euclideanDistance = dist(landmarkObs_obs.x, landmarkObs_obs.y, landmarkObs_pred.x, landmarkObs_pred.y);
      
      if(euclideanDistance < minDistance){
        minDistance = euclideanDistance;
        minIndex = landmarkObs_pred.id;
      }
      
    }
    // Update the min Index for that particular observation index
    observations[i].id = minIndex;
  }
  
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  
  for(int i = 0; i < num_particles; i++){
    vector<LandmarkObs> predicted;
    
    for(int j = 0; j < map_landmarks.landmark_list.size(); j++){
      double xDist = map_landmarks.landmark_list[j].x_f - particles[i].x;
      double yDist = map_landmarks.landmark_list[j].y_f - particles[i].y;
      if(fabs(xDist) <= sensor_range && fabs(yDist) <=sensor_range){
        predicted.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f});
      }
    }
    
    vector<LandmarkObs> translation;
    for (int k = 0; k < observations.size(); k++) {
      double xTrans = cos(particles[i].theta)*observations[k].x - sin(particles[i].theta)*observations[k].y + particles[i].x;
      double yTrans = sin(particles[i].theta)*observations[k].x + cos(particles[i].theta)*observations[k].y + particles[i].y;
      translation.push_back(LandmarkObs{ observations[k].id, xTrans, yTrans });
    }
    
    dataAssociation(predicted, translation);
    
    double xStd= std_landmark[0];
    double yStd= std_landmark[1];
    particles[i].weight = 1.0;
    for(int l=0; l<translation.size(); l++){
      double mu_x, mu_y;
      double x_obs = translation[l].x;
      double y_obs = translation[l].y;
      
      for(int m=0; m<predicted.size(); m++){
        if(predicted[m].id == translation[l].id){
          mu_x = predicted[m].x;
          mu_y = predicted[m].y;
        }
      }
      
      double gauss_norm= (1/(2 * M_PI * xStd * yStd));
      double exponent= ( pow(mu_x - x_obs, 2) / (2 * pow(xStd, 2)) + (pow(mu_y - y_obs, 2) / (2 * pow(yStd, 2))) );
      double localWeight = gauss_norm * exp(-exponent);
      particles[i].weight = particles[i].weight * localWeight ;
    }
    
  }
  
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  vector<double> weights;
  double maxWeight = -100000000.0;

// Calculating the max weights.
    for(int i = 0; i < num_particles; i++) {
        weights.push_back(particles[i].weight);
        if(particles[i].weight > maxWeight) {
            maxWeight = particles[i].weight;
        }
    }

    //resampling

    std::uniform_real_distribution<double> distDouble(0.0, maxWeight);
    std::uniform_int_distribution<int> distInt(0, num_particles - 1);
    int index = distInt(randomEngine);
    double beta = 0.0;
    vector<Particle> resampledParticles;
    for(int i = 0; i < num_particles; i++) {
        beta = beta + distDouble(randomEngine) * 2.0;
        while(beta > weights[index]) {
            beta = beta - weights[index];
            index = (index + 1) % num_particles;
        }
        resampledParticles.push_back(particles[index]);
    }

    particles = resampledParticles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}