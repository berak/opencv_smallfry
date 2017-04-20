#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

namespace ean {
namespace detail {
// transform [0..9] to 2bit error encoded 7bit luts
const vector<String> lh8       { "0001101", "0011001", "0010011", "0111101", "0100011", "0110001", "0101111", "0111011", "0110111", "0001011" };
const vector<String> rh8       { "1110010", "1100110", "1101100", "1000010", "1011100", "1001110", "1010000", "1000100", "1001000", "1110100" };
const vector<String> lh_odd13  { "0001101", "0011001", "0010011", "0111101", "0100011", "0110001", "0101111", "0111011", "0110111", "0001011" };
const vector<String> lh_even13 { "0100111", "0110011", "0011011", "0100001", "0011101", "0111001", "0000101", "0010001", "0001001", "0010111" };
const vector<String> rh_all13  { "1110010", "1100110", "1101100", "1000010", "1011100", "1001110", "1010000", "1000100", "1001000", "1110100" };
const vector<String> parity13  { "111111",  "110100",  "110010",  "110001",  "101100",  "100110",  "100011",  "101010",  "101001",  "100101"  };
// decode a single[0..9] digit
int dec( const String &bits, int off, const vector<String> &table, int howMany )
{
	String snip = bits.substr(off, howMany);
	for( int i=0; i<10; i++  )
	{
		if ( snip == table[i] )
			return i;
	}
	return -1;
}
int checkGuards( const String &bits, int end )
{
	int cen = end / 2;
	// left end
	if ( bits[0]     != '1' ) return 1; //"!L";
	if ( bits[1]     != '0' ) return 1; //"!L";
	if ( bits[2]     != '1' ) return 1; //"!L";
	// center marker
	if ( bits[cen-2] != '0' ) return 2; //"!C";
	if ( bits[cen-1] != '1' ) return 2; //"!C";
	if ( bits[cen  ] != '0' ) return 2; //"!C";
	if ( bits[cen+1] != '1' ) return 2; //"!C";
	if ( bits[cen+2] != '0' ) return 2; //"!C";
	// right end
	if ( bits[end-2] != '1' ) return 3; //"!E";
	if ( bits[end-1] != '0' ) return 3; //"!E";
	if ( bits[end  ] != '1' ) return 3; //"!E";

	return 0; // ok.
}
int checksum( int sum )
{
	int div = sum / 10;
	int rem = sum - div * 10;
	return ( rem ? 10 - rem : 0 );
}
int checksum8( const char v[] )
{
	return checksum(
		  (v[1]+v[3]+v[5])				// odd
		+ (v[0]+v[2]+v[4]+v[6]) * 3 );	// even
}
int checksum13( const char v[] )
{
	return checksum(
		(v[1]+v[3]+v[5]+v[7]+v[9]+v[11]) * 3	// odd
	  + (v[0]+v[2]+v[4]+v[6]+v[8]+v[10]) );		// even
}
String to_ean(char code[], int n) {
	String s="";
	for (int c=0; c<n; c++) {
		s += '0' + code[c];
	}
	return s;
}
Point quietZone(const String &line, int start, int stop, int step, int mean)
{
	Point res,tmp;
	for ( int i=start; (step>0&&i<stop)||(step<0&&i>stop); i+=step )
	{
		if ( line[i]> mean )
		{
			tmp.x = i;
			tmp.y ++;
		}
		else
		{
			if ( res.y < tmp.y )
				res.x = tmp.x, res.y=tmp.y;
			tmp.x=0;
			tmp.y=0;
		}
	}
	if ( res.y < tmp.y )
		res = tmp;
	return res;
}
} // detail
//
//
// public interface:
//
int decode8( const String &bits, String &ean )
{
	char code[8] = {0};
	int len= 67;
	int end = len-1;
	int ok = detail::checkGuards( bits, end );
	if ( ok != 0 )
		return 5;
	// decode the left 4 digits.
	int k=0, i=0,z=0,cen(len/2);
	for ( i=3; i<cen-2; i+=7,k++ )
	{
		// check for odd:
		z = detail::dec( bits, i, detail::lh8, 7 );
		if ( z == -1 )
		{
			return 6; //"!D";
		}
		code[k] = z;
	}
	// decode digits 4-8 including the checksum)
	for ( i=cen+3; i<end-2; i+=7,k++ )
	{
		z = detail::dec( bits, i, detail::rh8, 7 );
		if ( z == -1 )
		{
			return 7; //"!D";
		}
		code[k] = z;
	}
	// check again:
	int c = detail::checksum8( code );
	if ( code[7] != c )
	{
		//printf("!!    %s   checksum err : %d %d\n", code, c2i(code[7]),c );
		return 8;
	}
	ean=detail::to_ean(code,8);
	return 0; // ok!!!
}
int decode13( const String &bits, String &ean )
{
	char code[13] = {0};
	int len = 95, end = len-1, cen = len / 2;
	int ok  = detail::checkGuards( bits, end );
	if ( ok != 0 )
		return 5;
	// decode the left 6 digits(1-7).
	// they were encoded odd or even according to the parity digit(0),
	// store, which coding it was to reconstruct parity ( the first digit ) later.
	String par = "";
	int k = 0, i = 0, z = 0;
	for ( i=3; i<cen-2; i+=7,k++ )
	{
		// check for odd:
		z = detail::dec( bits, i, detail::lh_odd13, 7 );
		if ( z != -1 )
		{
			code[k+1] = z;
			par += '1';
			continue;
		}
		// check for even:
		z = detail::dec( bits, i, detail::lh_even13, 7 );
		if ( z != -1 )
		{
			code[k+1] = z;
			par += '0';
			continue;
		}
		// decode error !
		return 6; //"!D";
	}
	// decode parity( first digit ) from odd/even info collected before:
	code[0] = detail::dec( par, 0, detail::parity13, 6 );
	// decode digits 7-13 (including the checksum)
	for ( i=cen+3; i<end-2; i+=7,k++ )
	{
		z = detail::dec( bits, i, detail::rh_all13, 7 );
		if ( z == -1 )
		{
			return 7;
		}
		code[k+1] = z;
	}
	// check again:
	int c = detail::checksum13( code );
	if ( code[12] != c )
	{
		// kinda hard to let go here (after all that work!), but the result was wrong!
		return 8;
	}
	ean = detail::to_ean(code,13);
	return 0; // ok!!!
}
int sampleLine(const String &line, int y, String &code)
{
	double mean = 0;
	int i=0, x=0, w=line.size();
	// find quiet zones:
	Point sl = detail::quietZone(line, 0,   w/2,  1, mean );
	Point sr = detail::quietZone(line, w-1, w/2, -1, mean );
	int distance = sr.x - sl.x;
	if ( distance < 58 )
	{
		return 1; // too short
	}
	// try to sample a square wave from the grayscale data
	// between the quiet zones,
	// start at first black pixel
	bool last_state = true;
	vector<Point> pt(1,Point(0,0)); // pos,length, pos,length, ..
	for ( i = sl.x+1; i<sr.x; i++  )
	{
		// todo : proper nms
		bool state = (line[i] < mean);
		// toggle state:
		if ( state != last_state )
		{
			pt.push_back(Point(i, 1));
			last_state = state;
			continue;
		}
		// state unchanged, increase length of current section
		pt.back().y ++;
	}
	// ean13 has 95 bits and 58 state toggles,
	// ean8  has 67 bits and 42 state toggles.
	double siz = distance;
	if ( pt.size() == 58 )
	{
		siz /= 95.0; // ean13
	}
	else
	if ( pt.size() == 42 )
	{
		siz /= 67.0; // ean8
	}
	else
	{
		return 2; // no match to either ean8 or ean13
	}
	// scale pattern down to 1 pixel per bit:
	int sum = 0;
	double last = (sl.x+1);
	for( i=0; i<pt.size(); i++  )
	{
		double d = pt[i].x - last;
		last = pt[i].x;
		d = int( 0.5 + d/siz ); // floor ?
		sum += d;
		pt[i].y = d; // now holds bit count
	}
	if ( sum > 95 )
	{
		return 3; // too long
	}
	// make bitstring:
	bool bit = true;
	String bits = "";
	for( int i=0; i<pt.size(); i++ )
	{
		for( int j=0; j<pt[i].y; j++ )
		{
			bits += ( bit ? '1' : '0' );
			if ( bits.size() > 95 )
			{
				return 4; // still too long
			}
		}
		bit = ! bit;
	}
	bits += '1';
	if ( bits.size() == 67 )
		return decode8( bits, code );
	if ( bits.size() == 95 )
		return decode13( bits, code );
	return 4; // no match to either ean8 or ean13
}
} // ean

int main(int argc, char **argv)
{
    Mat A = Mat::eye(5,5,CV_32F);
    cerr << A << endl;
}
