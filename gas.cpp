
    #include "opencv2/ml.hpp"

    using namespace std;
    using namespace cv;

     
     
    class CV_EXPORTS NeuralGas : public CvStatModel {

        typedef float GasType;
        typedef vector<GasType> GasVec;

    public:
         struct CV_EXPORTS GasNode {
             int id;
             int rank;
             GasVec ref_vector;
             double distance;
         };

         //NeuralGas();
         virtual ~NeuralGas();

         // values taken from http://de.wikipedia.org/wiki/Neural_Gas
         NeuralGas( const Mat _distr, int _total_nodes, int _max_iterations, float _lambda0=1, float lambdaT=0.01f, float _epsilon0=.5f, float _epsilonT=0.001f );
         bool init();

         Mat cluster();
         bool train();
         bool train_auto();
         void clear();
         
         void get_nodes(std::vector<GasNode>& nodes) const;
         int get_iteration() const;
         int get_max_iterations() const;
         GasNode get_bmu() const;
         GasNode get_smu() const;
         GasNode get_wmu() const;
         //Scalar get_input() const;
         int get_total_nodes() const;
     
         // The networks settings are public so that should be changed dynamically depending the problem.
         float lambda0;
         float lambdaT;
         float epsilon0;
         float epsilonT;
     
    protected:
         std::vector<GasNode> nodes;
         GasNode bmu;
         GasNode smu;
         GasNode wmu;
         int total_nodes;
         int iteration, max_iterations;
     
//         Scalar input;
         Mat distribution;
     
         RNG rng;

    private:
         // Static method used for sorting the nodes.
         static bool Compare( const GasNode & node1, const GasNode & node2 ) {
             return ( node1.distance < node2.distance );
         }
    };

    //---8<-------------------------------------------------------------------------------------------------


    //NeuralGas::NeuralGas() {

    //}

    NeuralGas::NeuralGas( Mat _distr, int _total_nodes, int _max_iterations, float _lambda0, float _lambdaT, float _epsilon0, float _epsilonT ) {

        default_model_name = "neural_gas";

        distribution = _distr;
        total_nodes = _total_nodes;
        max_iterations = _max_iterations;
        iteration = 0;
        lambda0 = _lambda0 ? _lambda0 : _max_iterations/2;
        lambdaT = _lambdaT;
        epsilon0 = _epsilon0;
        epsilonT = _epsilonT;
      //  input = DBL_MAX;
        init();
    }

    NeuralGas::~NeuralGas() {
        clear( );
    }

    bool NeuralGas::init() {

        bool ok = true;

        int x = 0;
        int y = 0;

        // Create nodes.
        for( int i=0; i<total_nodes; i++ ) {
            NeuralGas::GasNode node;

            //x = rng.next() % (distribution.cols - 1);
            //y = rng.next() % (distribution.rows - 1);

            y = rng.next() % (distribution.rows - 1);            
            GasVec tmp_vector( distribution.row(y) );
            
            node.id = i;
            node.rank = 0;
            node.ref_vector = tmp_vector;
            node.distance = 0.0;

            nodes.push_back( node );
        }

        return ok;
    }

    bool NeuralGas::train_auto() {

        while( iteration < max_iterations ) {
            if( train() == false )
                return false;
        }

        return true;
    }

//    bool NeuralGas::train( Scalar _input ) {
    bool NeuralGas::train() {

        //if( _input[0] != DBL_MAX ) {
        //    input = _input;
        //} else {
            // peak random
            //int x = rng.next() % (distribution.cols - 1);
            int y = rng.next() % (distribution.rows - 1);

           // input = Scalar::all( distribution.at<float>(y,x) );
          GasVec input = distribution.row(y);
        //}

        // Calculate the distance of each node`s reference vector from the projected input vector.

        double temp = 0.0;
        double val = 0.0;

        for( int i=0; i<total_nodes; i++ ) {
            NeuralGas::GasNode & curr = nodes.at( i );

            curr.distance = 0.0;

            GasVec & ref_vector = curr.ref_vector;
            for( size_t x=0; x<ref_vector.size(); x++ ) {
                val = input[x] - ref_vector[x];
                //temp += pow( val, 2.0 );
                temp += val * val;
            }

            curr.distance = sqrt( temp );

            temp = 0.0;
            val = 0.0;
        }

        //Sort the nodes based on their distance.
        std::sort( nodes.begin(), nodes.end(), Compare);

        //Fetch the bmu/smu/wmu.
        bmu = nodes.at( 0 );
        smu = nodes.at( 1 );
        wmu = nodes.at( total_nodes - 1 );

        // Adapt the nodes.
        double epsilon_t = epsilon0 * pow( ( epsilonT / epsilon0 ), (float)iteration/max_iterations );
        double sqr_sigma = lambda0 * pow( ( lambdaT / lambda0 ), (float)iteration/max_iterations );

        for( int i=0; i<total_nodes; i++ ) {
            NeuralGas::GasNode & curr = nodes.at( i );
            curr.rank = -i;

            double h = exp( ((double)curr.rank) / sqr_sigma );

            GasVec & ref_vector = curr.ref_vector;

            for(size_t x=0; x<ref_vector.size(); x++){
                double delta = (input[x] - ref_vector[x]) * h * epsilon_t;
                ref_vector[x] += delta;
            }
        }

        iteration++;

        return true;
    }

    Mat NeuralGas::cluster() {
        Mat g;//(0,nodes[0].ref_vector.size(),CV_32F);
        for ( size_t n=0; n<nodes.size(); n++ ) {
            for ( size_t v=0; v<nodes[n].ref_vector.size(); v++ ) {
                g.push_back(nodes[n].ref_vector[v]);
            }
        }
        return g.reshape(1,nodes.size());
    }
    void NeuralGas::clear() {
        nodes.clear();
        distribution.release();
    }

    void NeuralGas::get_nodes(std::vector<GasNode>& _nodes) const {
        _nodes = nodes;
    }

    int NeuralGas::get_iteration() const {
        return iteration;
    }

    int NeuralGas::get_max_iterations() const {
        return max_iterations;
    }

    NeuralGas::GasNode NeuralGas::get_bmu() const {
        return bmu;
    }

    NeuralGas::GasNode NeuralGas::get_smu() const {
        return smu;
    }

    NeuralGas::GasNode NeuralGas::get_wmu() const {
        return wmu;
    }

    //Scalar NeuralGas::get_input() const {
    //    return input;
    //}

    int NeuralGas::get_total_nodes() const {
        return total_nodes;
    }

//---8<-------------------------------------------------------------------------------------------------

#include "opencv2/highgui.hpp"
#include "time.h"

    Mat draw(const Mat& dist) {
        Mat img(200,200,CV_8UC3);
        for ( int i=0; i<dist.rows; i++ ) {
            circle(img,Point2f(dist.at<float>(i,0),dist.at<float>(i,1)),2,Scalar(255),2);
        }
        return img;
    }

    int main() 
    {
        // whaa, having an empty constructor is probably a bad idea
 
        Mat dist(100,2,CV_32F);
        time_t now;
        time(&now);
        RNG rng(now);
        rng.fill(dist, RNG::UNIFORM,Scalar(),Scalar::all(200));

        NeuralGas gas(dist,9,200);
        
        imshow("pre",draw(dist));
        waitKey();

        for ( int i=0; i<200; i++ ) {
            if (! gas.train() )
                break;
            Mat dist2 = gas.cluster();

            imshow("post",draw(dist2));
            waitKey(40);
        }
        // so far, so well. and now ?
        // what do i train it on ? 
        // shouldn't ther be some kind of 'prediction' as well ?
        waitKey();
	    return 0;
    }
