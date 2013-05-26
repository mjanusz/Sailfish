#ifndef SLF_SUBDOMAIN_H
#define SLF_SUBDOMAIN_H 1

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include <cvmlcpp/base/Matrix>
#include <cvmlcpp/math/Euclid>
#include <cvmlcpp/volume/Geometry>
#include <cvmlcpp/volume/VolumeIO>
#include <cvmlcpp/volume/Voxelizer>
#include <cvmlcpp/volume/VoxelTools>

using namespace cvmlcpp;

typedef DTree<char, 3u> Octree;

iPoint3D NodeLocation(const Octree::DNode& node);
iPoint3D NodeExtent(const Octree::DNode& node);
int CountFluidNodes(const Octree::DNode& node);
void RemoveEmptyAreas(Octree::DNode node);
void FlushFluidCache();

extern const char kFluid;
extern const char kWall;

class Subdomain {
  public:
    Subdomain(iPoint3D origin, iPoint3D extent):
		origin_(origin), extent_(extent),
		fluid_nodes_(0)	{};

    Subdomain(iPoint3D origin, iPoint3D extent, int fluid_nodes):
		origin_(origin), extent_(extent),
		fluid_nodes_(fluid_nodes) {};

	Subdomain(const Octree::DNode node):
		origin_(NodeLocation(node)), extent_(NodeExtent(node)),
		fluid_nodes_(CountFluidNodes(node)) {};

	bool operator==(const Subdomain& rhs) const {
		return this->origin_ == rhs.origin_ &&
			this->extent_ == rhs.extent_ &&
			this->fluid_nodes_ == rhs.fluid_nodes_;
	}

	std::string JSON() const {
		std::ostringstream c;
		c << "{ pos: [" << origin_.x() << ", "
						<< origin_.y() << ", "
						<< origin_.z() << "], "
		  << "size: [" << extent_.x() - origin_.x() + 1 << ", "
		               << extent_.y() - origin_.y() + 1 << ", "
					   << extent_.z() - origin_.z() + 1 << "] }";
		return c.str();
	}

	// Builds the union of two subdomains.
	const Subdomain operator+(const Subdomain& rhs) const {
		Subdomain result = *this;
		result.origin_.set(
				std::min(result.origin_.x(), rhs.origin_.x()),
				std::min(result.origin_.y(), rhs.origin_.y()),
				std::min(result.origin_.z(), rhs.origin_.z()));
		result.extent_.set(
				std::max(result.extent_.x(), rhs.extent_.x()),
				std::max(result.extent_.y(), rhs.extent_.y()),
				std::max(result.extent_.z(), rhs.extent_.z()));
		result.fluid_nodes_ += rhs.fluid_nodes_;
		return result;
	}

	int len() const {
		return (extent_.x() - origin_.x() + 1);
	}

	// Returns the number of nodes contained within the subdomain.
	int volume() const {
		return (extent_.x() - origin_.x() + 1) *
			   (extent_.y() - origin_.y() + 1) *
			   (extent_.z() - origin_.z() + 1);
	}

	int fluid_nodes() const {
		return fluid_nodes_;
	}

	double fill_fraction() const {
		return static_cast<double>(fluid_nodes_) / volume();
	}

	friend std::ostream& operator<<(std::ostream& os, const Subdomain& s);

  private:
	iPoint3D origin_, extent_;  // location of the origin point and the point
							    // opposite to the origin
	int fluid_nodes_;
};

std::vector<Subdomain> ToSubdomains(const Octree::DNode node);

#endif
