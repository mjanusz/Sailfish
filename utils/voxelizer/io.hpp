// Simple logging for vectors.
template <typename T>
std::ostream& operator <<(std::ostream& os, const std::vector<T>& v) {
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(os));
    return os;
}

// Saves a dense matrix as a numpy array.
void SaveAsNumpy(const cvmlcpp::Matrix<char, 3u>& voxels, const std::string& fname) {
	std::ofstream out(fname + ".npy");
	out << "\x93NUMPY\x01";

	char buf[128] = {0};
	out.write(buf, 1);
	const std::size_t *ext = voxels.extents();
	if (voxels.colMajor()) {
		snprintf(buf, 128, "{'descr': 'bool', 'fortran_order': False, 'shape': (%lu, %lu, %lu)}",
				ext[2], ext[1], ext[0]);
	} else {
		snprintf(buf, 128, "{'descr': 'bool', 'fortran_order': False, 'shape': (%lu, %lu, %lu)}",
				ext[0], ext[1], ext[2]);
	}

	int i, len = strlen(buf);
	unsigned short int dlen = (((len + 10) / 16) + 1) * 16;

	for (i = 0; i < dlen - 10 - len; i++) {
		buf[len+i] = ' ';
	}
	buf[len+i] = 0x0;
	dlen -= 10;

	out.write((char*)&dlen, 2);
	out << buf;

	out.write(&(voxels.begin()[0]), voxels.size());
	out.close();
}
