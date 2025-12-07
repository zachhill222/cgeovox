#pragma once

#include <concepts>
#include <cstdint>
#include <string>
#include <bitset>
#include <bit>
#include <cmath>

namespace gv::util
{
	///////////////////////////////////////////////////////////////
	/// Union type for integer/float identification and manipulation
	///////////////////////////////////////////////////////////////
	template<int nBits>
		requires (nBits == 32 || nBits == 64) //add 128 later?
	union FloatingPointBits
	{
		///////////////////////////////////////////////////////////////
		/// Type defs
		///////////////////////////////////////////////////////////////
		static constexpr int BITS = nBits;

		using float_type = decltype([]() {
			if constexpr (BITS == 32) {return float{};}
			else if constexpr (BITS == 64) {return double{};}
			else static_assert(BITS == 32, "FloatingPointBits: unsuported number of bits");
		}());

		using int_type = decltype([]() {
			if constexpr (BITS == 32) {return int32_t{};}
			else if constexpr (BITS == 64) {return int64_t{};}
			else static_assert(BITS == 32, "FloatingPointBits: unsuported number of bits");
		}());


		using uint_type = decltype([]() {
			if constexpr (BITS == 32) {return uint32_t{};}
			else if constexpr (BITS == 64) {return uint64_t{};}
			else static_assert(BITS == 32, "FloatingPointBits: unsuported number of bits");
		}());

		///////////////////////////////////////////////////////////////
		/// Size of bit fields and exponent bias (IEEE 754)
		///////////////////////////////////////////////////////////////
		static constexpr int_type SIGN_BITS{1};
		static constexpr int_type MANTISSA_BITS = []() {
			if constexpr (BITS == 32) {return 23;}
			else if constexpr (BITS == 64) {return 52;}
			else static_assert(BITS == 32, "FloatingPointBits: unsuported number of bits");
		}();
		static constexpr int_type EXPONENT_BITS = []() {
			if constexpr (BITS == 32) {return 8;}
			else if constexpr (BITS == 64) {return 11;}
			else static_assert(BITS == 32, "FloatingPointBits: unsuported number of bits");
		}();
		static_assert(SIGN_BITS + MANTISSA_BITS + EXPONENT_BITS == BITS, "FloatingPointBits: inconsistent number of bits");

		static constexpr int_type EXPONENT_BIAS = []() {
			if constexpr (BITS == 32) {return 127;}
			else if constexpr (BITS == 64) {return 1023;}
			else static_assert(BITS == 32, "FloatingPointBits: unsuported number of bits");
		}();


		///////////////////////////////////////////////////////////////
		/// Get bit masks for each field
		///////////////////////////////////////////////////////////////
		static constexpr uint_type SIGN_MASK     =   uint_type(1) << (BITS - 1);                           //only need to shift 1 into place
		static constexpr uint_type EXPONENT_MASK = ((uint_type(1) << EXPONENT_BITS) - 1) << MANTISSA_BITS; //set correct number of ones, then shift into place
		static constexpr uint_type MANTISSA_MASK =  (uint_type(1) << MANTISSA_BITS) - 1;                    //only need to set the correct number of ones
		static_assert(SIGN_MASK + EXPONENT_MASK + MANTISSA_MASK == static_cast<uint_type>(-1), "FloatingPointBits: inconsistent masks");
		static_assert((SIGN_MASK & EXPONENT_MASK) == 0, "FloatingPointBits: inconsistent masks");
		static_assert((SIGN_MASK & MANTISSA_MASK) == 0, "FloatingPointBits: inconsistent masks");
		static_assert((MANTISSA_MASK & EXPONENT_MASK) == 0, "FloatingPointBits: inconsistent masks");

		///////////////////////////////////////////////////////////////
		/// Union fields
		///////////////////////////////////////////////////////////////
		float_type f;
		uint_type  i;

		///////////////////////////////////////////////////////////////
		/// Constructors
		///////////////////////////////////////////////////////////////
		constexpr FloatingPointBits() noexcept : i(0) {};
		constexpr explicit FloatingPointBits(float_type val) noexcept : f(val) {};
		constexpr explicit FloatingPointBits(uint_type  val) noexcept : i(val) {};

		///////////////////////////////////////////////////////////////
		/// Bit setting
		///////////////////////////////////////////////////////////////
		constexpr void set_sign(const bool negative) noexcept
		{
			if (negative) {i |= SIGN_MASK;}
			else {i &= ~SIGN_MASK;}
		}
		
		constexpr void set_exponent(const uint_type exp) noexcept
		{
			i &= ~EXPONENT_MASK; //clear bits
			i |= (exp << MANTISSA_BITS) & EXPONENT_MASK; //shift bits into place and set
		}

		constexpr void set_exponent_from_power_of_2(const int_type power) noexcept
		{
			set_exponent(power + EXPONENT_BIAS);
		}

		constexpr void set_mantissa(const uint_type mantissa) noexcept
		{
			i &= ~MANTISSA_MASK; //clear bits
			i |= mantissa & MANTISSA_MASK; //set bits
		}

		///////////////////////////////////////////////////////////////
		/// Bit views
		///////////////////////////////////////////////////////////////
		constexpr uint_type sign() const noexcept
		{
			return (i & SIGN_MASK) >> BITS - 1;
		}

		// extract or set the exponent as it appears in the bits
		constexpr uint_type exponent_raw() const noexcept
		{
			return (i & EXPONENT_MASK) >> MANTISSA_BITS;
		}
		

		// extract exponent as the power for scientific notation
		constexpr int64_t exponent_actual() const noexcept
		{
			return static_cast<int64_t>(exponent_raw()) - EXPONENT_BIAS;
		}

		// extract or set the mantissa bits
		constexpr uint_type mantissa() const noexcept
		{
			return i & MANTISSA_MASK;
		}
		

		// get bits as a string for debugging
		std::string to_string() const noexcept
		{
			std::string raw = std::bitset<BITS>(i).to_string();
			std::string result;
			result += raw[0];
			result += "|" + raw.substr(SIGN_BITS, EXPONENT_BITS);
			result += "|" + raw.substr(SIGN_BITS + EXPONENT_BITS, MANTISSA_BITS);
			return result;
		}
	};


	///////////////////////////////////////////////////////////////
	/// This class provides exact arithmetic in fixed point precision.
	/// All numbers are stored as 
	///          mantissa * 2^exponent
	/// where exponent is an integer power known at compile time
	/// and mantissa is an integer that stores the current value.
	/// Similar to floating point numbers,
	///          exponent = bias - fraction_bits
	/// but unlike floating point numbers, this is constant for all
	/// numbers of this type. The fraction_bits are half the size of the
	/// integer (mantissa) type and the bias/offset is 0 by default
	///////////////////////////////////////////////////////////////

	template<std::signed_integral Mantissa_t = int64_t, int OFFSET=0>
	struct FixedPoint
	{
		///////////////////////////////////////////////////////////////
		/// Type defs
		///////////////////////////////////////////////////////////////
		/// allow Scalar concept to see this
		static constexpr bool IS_SCALAR = true;

		/// Underlying integer type
		using mantissa_type = Mantissa_t;

		/// Floating point type with the same number of bits
		using Float_t   = decltype([]() {
			if constexpr (sizeof(Mantissa_t) <= 4) {return float{};}
			else if constexpr (sizeof(Mantissa_t) <= 8) {return double{};}
			else {return (long double){};}
		}());

		/// Intermediate integer type with double the bits for exact multiplications
		using Intermediate_t = decltype([]() {
			if constexpr (sizeof(Mantissa_t) <= 2) {return int32_t{};}
			else if constexpr (sizeof(Mantissa_t) <= 4) {return int64_t{};}
			#ifdef __SIZEOF_INT128__ //should work on most compilers/machines. maybe not windows?
				else {return __int128{};}
			#else
				else {
					static_assert(sizeof(Mantissa_t) <= 8, "FixedPoint: 128-bit arithmetic not available (__SIZEOF_INT128__ not defined)");
					return int64_t{};
				}
			#endif
		}());

		/// Unsigned intermediate type for bit shifting
		using uIntermediate_t = decltype([]() {
			if constexpr (sizeof(Mantissa_t) <= 2) {return uint32_t{};}
			else if constexpr (sizeof(Mantissa_t) <= 4) {return uint64_t{};}
			#ifdef __SIZEOF_INT128__ //should work on most compilers/machines. maybe not windows?
				else {return (unsigned __int128){};}
			#else
				else {
					static_assert(sizeof(Mantissa_t) <= 8, "FixedPoint: 128-bit arithmetic not available (__SIZEOF_INT128__ not defined)");
					return uint64_t{};
				}
			#endif
		}());

		/// Conversion type to help cast to float or double
		using Conversion_t = FloatingPointBits<sizeof(Mantissa_t)*8>;

		///////////////////////////////////////////////////////////////
		/// Current value and exponent
		///////////////////////////////////////////////////////////////
		Mantissa_t mantissa;
		static constexpr Mantissa_t FRACTION_BITS = Mantissa_t{sizeof(Mantissa_t) * 4}; //half the bits
		static constexpr Mantissa_t EXPONENT      = Mantissa_t{OFFSET} - FRACTION_BITS;

		static constexpr Float_t SCALE = []() {
			//set the exponent
			typename Conversion_t::uint_type bits = static_cast<typename Conversion_t::uint_type>(EXPONENT + Conversion_t::EXPONENT_BIAS);
			//shift into place (implicit leading 1)
			bits <<= Conversion_t::MANTISSA_BITS;
			return std::bit_cast<Float_t>(bits);
		}();

		static constexpr Float_t INV_SCALE = []() {
			//set the exponent
			typename Conversion_t::uint_type bits = static_cast<typename Conversion_t::uint_type>(-EXPONENT + Conversion_t::EXPONENT_BIAS);
			//shift into place (implicit leading 1)
			bits <<= Conversion_t::MANTISSA_BITS;
			return std::bit_cast<Float_t>(bits);
		}();

		static constexpr Float_t EPSILON = SCALE; //the scale is also the  spacing between consecutive numbers
		static constexpr Float_t MAX_FLOAT = Float_t(static_cast<Float_t>(std::numeric_limits<Mantissa_t>::max()) * SCALE);

		//overflow if a positive intermediate calculation is larger than this
		//intermediate calculations are done in a larger unsigned type for easier bit shifting
		//this bound is the same if the final result is positive or negative
		static constexpr uIntermediate_t MAX_MANTISSA = static_cast<uIntermediate_t>(std::numeric_limits<Mantissa_t>::max());

		///////////////////////////////////////////////////////////////
		/// Constructors
		///////////////////////////////////////////////////////////////
		constexpr FixedPoint() noexcept : mantissa(0) {}
		constexpr FixedPoint(const std::floating_point auto x) noexcept : mantissa(static_cast<Mantissa_t>(std::llround(static_cast<Float_t>(x) * INV_SCALE))) {assert(std::fabs(x) <= MAX_FLOAT);} //scale and round
		constexpr FixedPoint(const std::integral auto x)       noexcept : FixedPoint(static_cast<Float_t>(x)) {}
		constexpr explicit FixedPoint(const Mantissa_t x, int) noexcept : mantissa(x) {} //tagged constructor to explicitly set the mantissa

		///////////////////////////////////////////////////////////////
		/// Conversions
		///////////////////////////////////////////////////////////////
		constexpr operator Float_t() const noexcept
		{
			if (mantissa==0) {return Float_t(0);}

			Conversion_t conv;

			//set sign
			const bool negative = mantissa<0;
			conv.set_sign(negative);

			//get exponent after shifting decimal place to 1.float_mantissa * 2^float_exponent
			typename Conversion_t::uint_type abs_mantissa = negative ? -mantissa : mantissa;
			const int leading_bit_pos = Conversion_t::BITS - 1 - std::countl_zero(abs_mantissa);
			const typename Conversion_t::int_type float_exponent = EXPONENT + leading_bit_pos;
			conv.set_exponent_from_power_of_2(float_exponent);

			//get mantissa bits (the leading 1 is ignored when using conv.set_mantissa)
			if (leading_bit_pos >= Conversion_t::MANTISSA_BITS)
			{
				//shift right
				conv.set_mantissa(abs_mantissa >> (leading_bit_pos - Conversion_t::MANTISSA_BITS));
			}
			else 
			{
				//shift left
				conv.set_mantissa(abs_mantissa << (Conversion_t::MANTISSA_BITS - leading_bit_pos));
			}

			return conv.f;
		}

		///////////////////////////////////////////////////////////////
		/// Arithmetic
		///////////////////////////////////////////////////////////////
		constexpr FixedPoint  operator+( const FixedPoint &other) noexcept {return FixedPoint(mantissa + other.mantissa, 0);}
		constexpr FixedPoint& operator+=(const FixedPoint &other) noexcept {mantissa += other.mantissa; return *this;}
		constexpr FixedPoint  operator-( const FixedPoint &other) noexcept {return FixedPoint(mantissa - other.mantissa, 0);}
		constexpr FixedPoint& operator-=(const FixedPoint &other) noexcept {mantissa -= other.mantissa; return *this;}
		
		constexpr FixedPoint  operator*(const FixedPoint &other)  noexcept
		{
			// (m1 * 2^EXP) * (m2 * 2^EXP) = m3 * 2^EXP where m3 = m1*m2*2^EXP
			// we can set m3 = (m1*m2 << EXP) if EXP>0.
			
			// Determine signs
			bool num_negative = mantissa < 0;
			bool den_negative = other.mantissa < 0;
			bool result_negative = num_negative != den_negative;

			// Work with absolute values
			uIntermediate_t unum = num_negative ? static_cast<uIntermediate_t>(-mantissa) : static_cast<uIntermediate_t>(mantissa);
			uIntermediate_t uden = den_negative ? static_cast<uIntermediate_t>(-other.mantissa) : static_cast<uIntermediate_t>(other.mantissa);

			// Perform division with unsigned values and apply shift
			uIntermediate_t uval = unum * uden;
			if constexpr (EXPONENT > 0) {uval <<= EXPONENT;}
			else {uval >>= -EXPONENT;}

			//check for overflow
			assert(uval<MAX_MANTISSA);

			// Get final result
			Mantissa_t m_val = result_negative ? -static_cast<Mantissa_t>(uval) : static_cast<Mantissa_t>(uval);
			return FixedPoint(m_val, 0);
		}
		constexpr FixedPoint& operator*=(const FixedPoint &other) noexcept
		{
			// Determine signs
			bool num_negative = mantissa < 0;
			bool den_negative = other.mantissa < 0;
			bool result_negative = num_negative != den_negative;

			// Work with absolute values
			uIntermediate_t unum = num_negative ? static_cast<uIntermediate_t>(-mantissa) : static_cast<uIntermediate_t>(mantissa);
			uIntermediate_t uden = den_negative ? static_cast<uIntermediate_t>(-other.mantissa) : static_cast<uIntermediate_t>(other.mantissa);

			// Perform division with unsigned values and apply shift
			uIntermediate_t uval = unum * uden;
			if constexpr (EXPONENT > 0) {uval <<= EXPONENT;}
			else {uval >>= -EXPONENT;}

			//check for overflow
			assert(uval<MAX_MANTISSA);

			// Get final result
			mantissa = result_negative ? -static_cast<Mantissa_t>(uval) : static_cast<Mantissa_t>(uval);
			return *this;
		}

		constexpr FixedPoint  operator/(const FixedPoint &other)  noexcept
		{
			// (m1 / 2^EXP) * (m2 / 2^EXP) = m3 * 2^EXP where m3 = (m1 / m2) * 2^-EXP = ((m1 * 2^M) / m2 )* 2^(-M-EXP) 
			// where we choose M to take advantage of the full intermediate precision
			constexpr int M  = 4*sizeof(Intermediate_t);
			constexpr int final_shift = M + EXPONENT; //negative of what is in the comment
			static_assert(M == 8*sizeof(Mantissa_t));

			// Determine signs
			const bool num_negative    = mantissa < 0;
			const bool den_negative    = other.mantissa < 0;
			const bool result_negative = num_negative != den_negative;

			// Work with absolute values
			const uIntermediate_t unum = num_negative ? static_cast<uIntermediate_t>(-mantissa) : static_cast<uIntermediate_t>(mantissa);
			const uIntermediate_t uden = den_negative ? static_cast<uIntermediate_t>(-other.mantissa) : static_cast<uIntermediate_t>(other.mantissa);

			// Perform division with unsigned values and apply shift
			uIntermediate_t uval = (unum << M) / uden;
			if constexpr (final_shift > 0) {uval >>= final_shift;}
			else {uval <<= -final_shift;}

			//check for overflow
			assert(uval<MAX_MANTISSA);

			//convert back to signed integer and convert to mantissa
			Mantissa_t m_val = result_negative ? -static_cast<Mantissa_t>(uval) : static_cast<Mantissa_t>(uval);
			return FixedPoint(m_val, 0);
		}
		constexpr FixedPoint& operator/=(const FixedPoint &other) noexcept
		{
			constexpr int M  = 4*sizeof(Intermediate_t);
			constexpr int final_shift = M + EXPONENT; //negative of what is in the comment
			static_assert(M == 8*sizeof(Mantissa_t));

			// Determine signs
			const bool num_negative    = mantissa < 0;
			const bool den_negative    = other.mantissa < 0;
			const bool result_negative = num_negative != den_negative;

			// Work with absolute values
			const uIntermediate_t unum = num_negative ? static_cast<uIntermediate_t>(-mantissa) : static_cast<uIntermediate_t>(mantissa);
			const uIntermediate_t uden = den_negative ? static_cast<uIntermediate_t>(-other.mantissa) : static_cast<uIntermediate_t>(other.mantissa);

			// Perform division with unsigned values and apply shift
			uIntermediate_t uval = (unum << M) / uden;
			if constexpr (final_shift > 0) {uval >>= final_shift;}
			else {uval <<= -final_shift;}

			//check for overflow
			assert(uval<MAX_MANTISSA);

			//convert back to signed integer and convert to mantissa
			mantissa = result_negative ? -static_cast<Mantissa_t>(uval) : static_cast<Mantissa_t>(uval);
			return *this;
		}

		///////////////////////////////////////////////////////////////
		/// Extra Math
		///////////////////////////////////////////////////////////////
		constexpr FixedPoint operator-() const noexcept {return FixedPoint(-mantissa,0);}

		///////////////////////////////////////////////////////////////
		/// FixedPoint Comparisons
		///////////////////////////////////////////////////////////////
		constexpr bool operator<(const FixedPoint &other)  const noexcept {return mantissa  < other.mantissa;}
		constexpr bool operator<=(const FixedPoint &other) const noexcept {return mantissa <= other.mantissa;}
		constexpr bool operator>(const FixedPoint &other)  const noexcept {return mantissa  > other.mantissa;}
		constexpr bool operator>=(const FixedPoint &other) const noexcept {return mantissa >= other.mantissa;}

		constexpr bool operator==(const FixedPoint &other) const noexcept {return mantissa == other.mantissa;}
		constexpr bool operator!=(const FixedPoint &other) const noexcept {return mantissa != other.mantissa;}

		///////////////////////////////////////////////////////////////
		/// Integer Comparisons
		///////////////////////////////////////////////////////////////
		// constexpr bool operator<(const Mantissa_t &other)  const noexcept {return *this < FixedPoint(other);}
		// constexpr bool operator<=(const Mantissa_t &other) const noexcept {return *this <= FixedPoint(other);}
		// constexpr bool operator>(const Mantissa_t &other)  const noexcept {return *this > FixedPoint(other);}
		// constexpr bool operator>=(const Mantissa_t &other) const noexcept {return *this >= FixedPoint(other);}

		// constexpr bool operator==(const Mantissa_t &other) const noexcept {return *this == FixedPoint(other);}
		// constexpr bool operator!=(const Mantissa_t &other) const noexcept {return *this != FixedPoint(other);}

		///////////////////////////////////////////////////////////////
		/// Float Comparisons
		///////////////////////////////////////////////////////////////
		constexpr bool operator<(const Float_t &other)  const noexcept {return *this < FixedPoint(other);}
		constexpr bool operator<=(const Float_t &other) const noexcept {return *this <= FixedPoint(other);}
		constexpr bool operator>(const Float_t &other)  const noexcept {return *this > FixedPoint(other);}
		constexpr bool operator>=(const Float_t &other) const noexcept {return *this >= FixedPoint(other);}

		constexpr bool operator==(const Float_t &other) const noexcept {return *this == FixedPoint(other);}
		constexpr bool operator!=(const Float_t &other) const noexcept {return *this != FixedPoint(other);}



		// get bits as a string for debugging
		std::string to_string() const noexcept
		{
			std::string raw = std::bitset<Conversion_t::BITS>(std::abs(mantissa)).to_string();
			std::string result;
			result += std::to_string(mantissa<0); //get correct sign bit
			result += "|" + raw.substr(1, Conversion_t::BITS - 1);
			result += " * 2^" + std::to_string(EXPONENT);
			return result;
		}
	};
}