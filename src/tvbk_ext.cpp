#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include "tvbk.hpp"

namespace nb = nanobind;
using namespace nb::literals;
typedef nb::ndarray<float, nb::numpy, nb::device::cpu, nb::shape<-1>, nb::c_contig> fvec;
typedef nb::ndarray<float, nb::numpy, nb::device::cpu, nb::shape<-1,-1>, nb::c_contig> fmat;
typedef nb::ndarray<float, nb::numpy, nb::device::cpu, nb::shape<-1,-1,-1>, nb::c_contig> farr3;
typedef nb::ndarray<float, nb::numpy, nb::device::cpu, nb::shape<-1,-1,-1,-1>, nb::c_contig> farr4;
typedef nb::ndarray<uint32_t, nb::numpy, nb::device::cpu, nb::shape<-1>, nb::c_contig> uvec;

template<typename shape> 
using farr = nb::ndarray<float, nb::numpy, nb::device::cpu, shape, nb::c_contig>;

// boilerplate for declaring a model stepping function
template <typename model, typename M, int width=8> void decl_step(M m)
{
  char name[64];
  snprintf(name, 64, "step_%s", model::name);
  m.def(name,
    [](const tvbk::cxbs<width> &cx, const tvbk::conn &c,
    farr<nb::shape<-1,model::num_svar,-1,width>> &x,
    farr<nb::shape<-1,model::num_svar,-1,width>> &y,
    farr<nb::shape<-1,model::num_svar,width>> &z,
    farr<nb::shape<-1,-1,model::num_parm,width>> &p,
    uint32_t t0, uint32_t nt, float dt,
    nb::ndarray<uint64_t, nb::numpy, nb::device::cpu, nb::shape<-1, width, 4>> seed
  )
    {
      auto fail = [&](const char *msg) {
#ifdef __EMSCRIPTEN__
        char log_cmd[256];
        snprintf(log_cmd, 256, "console.error('%s')", msg);
        emscripten_run_script(log_cmd);
#else
        throw std::runtime_error(msg);
#endif
      };

      // check x.shape[0] == p.shape[0] == cx.num_batch
      if (!(x.shape(0) == p.shape(0) && x.shape(0) == cx.num_batch))
      {
        return fail("batch shapes don't match: check that x.shape[0] == p.shape[0] == cx.num_batch");
      }
      // check x.shape == y.shape
      if (!( x.shape(0) == y.shape(0) && x.shape(2) == y.shape(2) ))
      {
        return fail("x and y shapes do not match!");
      }
      // check z.shape
      if (!( x.shape(0) == z.shape(0) && seed.shape(0) == x.shape(0)))
      {
        return fail("z.shape or seed.shape doesn't match x");
      }
      // check x.shape[2] == cx.num_node == p.shape[1]
      if (!(x.shape(2) == cx.num_node && x.shape(2)))
      {
        return fail("node shapes don't match: check that x.shape[2] == cx.num_node");
      }
      if (!(p.shape(1) == cx.num_node || p.shape(1) == 1))
      {
        return fail("p.shape[1] must be num_node or 1.");
      }
      tvbk::step_batches<model, 8>(cx, c, (float *)x.data(), (float *)y.data(),
                                   (float *)z.data(), (float *)p.data(),
                                   p.shape(1) == cx.num_node, t0, nt, dt,
                                   (uint64_t *)seed.data());
    }, "cx"_a, "c"_a, "x"_a, "y"_a, "z"_a, "p"_a, "t0"_a, "nt"_a, "dt"_a, "seed"_a);
}

// SFINAE helper to check for default_parms
template <typename T, typename = void>
struct has_defaults : std::false_type {};

template <typename T>
struct has_defaults<T, std::void_t<decltype(T::default_parms)>> : std::true_type {};

template <typename model, int width=8>
void bind_model(nb::module_ &m, const char* name) {
    auto cls = nb::class_<model>(m, name);
    
    cls.def_ro_static("num_svar", &model::num_svar)
       .def_ro_static("num_parm", &model::num_parm)
       .def_ro_static("num_cvar", &model::num_cvar)
       .def_ro_static("name", &model::name)
       .def_ro_static("parms", &model::parms);

    if constexpr (has_defaults<model>::value) {
        cls.def_prop_ro_static("default_parms", [](nb::object) {
             float *data = new float[model::num_parm];
             for(size_t i=0; i<model::num_parm; ++i) data[i] = model::default_parms[i];
             
             nb::capsule owner(data, [](void *p) noexcept { delete[] (float *)p; });
             return fvec(data, {model::num_parm}, owner);
        });
    }

    cls.def_static("dfun", 
        [](farr<nb::shape<model::num_svar, width>> &dx,
           farr<nb::shape<model::num_svar, width>> &x,
           farr<nb::shape<model::num_cvar, width>> &c,
           farr<nb::shape<model::num_parm, width>> &p)
        {  
          model::template dfun<width>((float *)dx.data(), (float *)x.data(), (float *)c.data(), (float *)p.data());
        }, "dx"_a, "x"_a, "c"_a, "p"_a
    );

    cls.def_static("step",
        [](const tvbk::cxbs<width> &cx, const tvbk::conn &c,
        farr<nb::shape<-1,model::num_svar,-1,width>> &x,
        farr<nb::shape<-1,model::num_svar,-1,width>> &y,
        farr<nb::shape<-1,model::num_svar,width>> &z,
        farr<nb::shape<-1,-1,model::num_parm,width>> &p,
        uint32_t t0, uint32_t nt, float dt,
        nb::ndarray<uint64_t, nb::numpy, nb::device::cpu, nb::shape<-1, width, 4>> seed
        )
        {
          auto fail = [&](const char *msg) {
    #ifdef __EMSCRIPTEN__
            char log_cmd[256];
            snprintf(log_cmd, 256, "console.error('%s')", msg);
            emscripten_run_script(log_cmd);
    #else
            throw std::runtime_error(msg);
    #endif
          };

          if (!(x.shape(0) == p.shape(0) && x.shape(0) == cx.num_batch))
            return fail("batch shapes don't match: check that x.shape[0] == p.shape[0] == cx.num_batch");
          if (!( x.shape(0) == y.shape(0) && x.shape(2) == y.shape(2) ))
            return fail("x and y shapes do not match!");
          if (!( x.shape(0) == z.shape(0) && seed.shape(0) == x.shape(0)))
            return fail("z.shape or seed.shape doesn't match x");
          if (!(x.shape(2) == cx.num_node && x.shape(2)))
            return fail("node shapes don't match: check that x.shape[2] == cx.num_node");
          if (!(p.shape(1) == cx.num_node || p.shape(1) == 1))
            return fail("p.shape[1] must be num_node or 1.");
          
          tvbk::step_batches<model, width>(cx, c, (float *)x.data(), (float *)y.data(),
                                       (float *)z.data(), (float *)p.data(),
                                       p.shape(1) == cx.num_node, t0, nt, dt,
                                       (uint64_t *)seed.data());
        }, "cx"_a, "c"_a, "x"_a, "y"_a, "z"_a, "p"_a, "t0"_a, "nt"_a, "dt"_a, "seed"_a
    );
}

NB_MODULE(tvbk_ext, m) {

  m.def(
      "randn",
      [](const uint32_t seed, fvec z) {
        uint64_t s[4] = {seed, seed, seed, seed};
        tvbk::randn(s, z.shape(0), (float *)z.data());
      },
      "seed"_a, "z"_a,
      "This functions generates normally distributed random numbers using a "
      "popcount trick.");

  nb::class_<tvbk::cx>(m, "Cx")
      .def(nb::init<uint32_t, uint32_t>(), "num_node"_a, "num_time"_a)
      .def_ro("num_node", &tvbk::cx::num_node)
      .def_ro("num_time", &tvbk::cx::num_time)
      .def_prop_ro("buf",
                   [](nb::object obj) {
                     auto &cx = nb::cast<tvbk::cx &>(obj);
                     return fmat(cx.buf, {cx.num_node, cx.num_time}, obj);
                   })
      .def_prop_ro("cx1",
                   [](nb::object obj) {
                     auto &cx = nb::cast<tvbk::cx &>(obj);
                     return fvec(cx.cx1, {cx.num_node}, obj);
                   })
      .def_prop_ro("cx2",
                   [](nb::object obj) {
                     auto &cx = nb::cast<tvbk::cx &>(obj);
                     return fvec(cx.cx2, {cx.num_node}, obj);
                   });

  nb::class_<tvbk::cx8>(m, "Cx8")
      .def(nb::init<uint32_t, uint32_t>(), "num_node"_a, "num_time"_a)
      .def_ro("num_node", &tvbk::cx8::num_node)
      .def_ro("num_time", &tvbk::cx8::num_time)
      .def_ro("num_item", &tvbk::cx8::num_item)
      .def_prop_ro("buf",
                   [](nb::object obj) {
                     auto &cx = nb::cast<tvbk::cx8 &>(obj);
                     return fmat(cx.buf, {cx.num_node, cx.num_time, cx.num_item}, obj);
                   })
      .def_prop_ro("cx1",
                   [](nb::object obj) {
                     auto &cx = nb::cast<tvbk::cx8 &>(obj);
                     return fmat(cx.cx1, {cx.num_node, cx.num_item}, obj);
                   })
      .def_prop_ro("cx2",
                   [](nb::object obj) {
                     auto &cx = nb::cast<tvbk::cx8 &>(obj);
                     return fmat(cx.cx2, {cx.num_node, cx.num_item}, obj);
                   });

  nb::class_<tvbk::cx8s>(m, "Cx8s")
      .def(nb::init<uint32_t, uint32_t, uint32_t>(), "num_node"_a, "num_time"_a, "num_batches"_a)
      .def_ro("num_node", &tvbk::cx8s::num_node)
      .def_ro("num_time", &tvbk::cx8s::num_time)
      .def_ro("num_item", &tvbk::cx8s::num_item)
      .def_ro("num_batch", &tvbk::cx8s::num_batch)
      .def_prop_ro("buf",
                   [](nb::object obj) {
                     auto &cx = nb::cast<tvbk::cx8s &>(obj);
                     return farr4(cx.buf, {cx.num_batch, cx.num_node, cx.num_time, cx.num_item}, obj);
                   })
      .def_prop_ro("cx1",
                   [](nb::object obj) {
                     auto &cx = nb::cast<tvbk::cx8s &>(obj);
                     return farr3(cx.cx1, {cx.num_batch, cx.num_node, cx.num_item}, obj);
                   })
      .def_prop_ro("cx2",
                   [](nb::object obj) {
                     auto &cx = nb::cast<tvbk::cx8s &>(obj);
                     return farr3(cx.cx2, {cx.num_batch, cx.num_node, cx.num_item}, obj);
                   })
    ;

  nb::class_<tvbk::conn>(m, "Conn")
      .def(nb::init<uint32_t, uint32_t>(), "num_node"_a, "num_nonzero"_a)
      .def_ro("num_node", &tvbk::conn::num_node)
      .def_ro("num_nonzero", &tvbk::conn::num_nonzero)
      .def_prop_ro("weights",
                   [](nb::object obj) {
                     auto &c = nb::cast<tvbk::conn &>(obj);
                     return fvec(const_cast<float *>(c.weights),
                                 {c.num_nonzero}, obj);
                   })
      .def_prop_ro("indices",
                   [](nb::object obj) {
                     auto &c = nb::cast<tvbk::conn &>(obj);
                     return uvec(const_cast<uint32_t *>(c.indices),
                                 {c.num_nonzero}, obj);
                   })
      .def_prop_ro("indptr",
                   [](nb::object obj) {
                     auto &c = nb::cast<tvbk::conn &>(obj);
                     return uvec(const_cast<uint32_t *>(c.indptr),
                                 {c.num_node + 1}, obj);
                   })
      .def_prop_ro("idelays", [](nb::object obj) {
        auto &c = nb::cast<tvbk::conn &>(obj);
        return uvec(const_cast<uint32_t *>(c.idelays), {c.num_nonzero}, obj);
      });

  m.def(
      "cx_j",
      [](const tvbk::cx &cx, const tvbk::conn &conn, uint32_t t) {
        tvbk::cx_j(cx, conn, t);
      },
      "cx"_a, "conn"_a, "t"_a,
      "This function calculates the afferent coupling buffer.");

  m.def(
      "cx_j8",
      [](const tvbk::cx8 &cx8, const tvbk::conn &conn, uint32_t t) {
        tvbk::cx_j_b<8>(cx8, conn, t);
      },
      "cx8"_a, "conn"_a, "t"_a,
      "This function calculates batched afferent coupling buffer.");

  m.def("cxs8_j",
        [](const tvbk::cx8s &cxs8, const tvbk::conn &conn, uint32_t t) {
          tvbk::cx_j_bs<8>(cxs8, conn, t);
        },
        "cxs8"_a, "conn"_a, "t"_a,
        "This function calculates many batched afferent coupling buffer.");

  m.def("dfun_jr8",
    [](farr<nb::shape<tvbk::jr::num_svar,8>> &dx,
       farr<nb::shape<tvbk::jr::num_svar,8>> &x,
       farr<nb::shape<tvbk::jr::num_cvar,8>> &c,
       farr<nb::shape<tvbk::jr::num_parm,8>> &p)
    {  
      tvbk::jr::dfun<8>((float *)dx.data(), (float *)x.data(), (float *)c.data(), (float *)p.data());
    }, "dx"_a, "x"_a, "c"_a, "p"_a);

  m.def("dfun_mpr8",
    [](farr<nb::shape<tvbk::mpr::num_svar,8>> &dx,
       farr<nb::shape<tvbk::mpr::num_svar,8>> &x,
       farr<nb::shape<tvbk::mpr::num_cvar,8>> &c,
       farr<nb::shape<tvbk::mpr::num_parm,8>> &p)
    {  
      tvbk::mpr::dfun<8>((float *)dx.data(), (float *)x.data(), (float *)c.data(), (float *)p.data());
    }, "dx"_a, "x"_a, "c"_a, "p"_a);

  m.def("dfun_kionex8",
    [](farr<nb::shape<tvbk::kionex::num_svar,8>> &dx,
       farr<nb::shape<tvbk::kionex::num_svar,8>> &x,
       farr<nb::shape<tvbk::kionex::num_cvar,8>> &c,
       farr<nb::shape<tvbk::kionex::num_parm,8>> &p)
    {  
      tvbk::kionex::dfun<8>((float *)dx.data(), (float *)x.data(), (float *)c.data(), (float *)p.data());
    }, "dx"_a, "x"_a, "c"_a, "p"_a);

  m.def("dfun_kionex2_8",
    [](farr<nb::shape<tvbk::kionex2::num_svar,8>> &dx,
       farr<nb::shape<tvbk::kionex2::num_svar,8>> &x,
       farr<nb::shape<tvbk::kionex2::num_cvar,8>> &c,
       farr<nb::shape<tvbk::kionex2::num_parm,8>> &p)
    {  
      tvbk::kionex2::dfun<8>((float *)dx.data(), (float *)x.data(), (float *)c.data(), (float *)p.data());
    }, "dx"_a, "x"_a, "c"_a, "p"_a);

  decl_step<tvbk::jr>(m);
  decl_step<tvbk::mpr>(m);
  decl_step<tvbk::kionex>(m);
  decl_step<tvbk::kionex2>(m);

  bind_model<tvbk::jr>(m, "JR");
  bind_model<tvbk::mpr>(m, "MPR");
  bind_model<tvbk::kionex>(m, "Kionex");
  bind_model<tvbk::kionex2>(m, "Kionex2");
}
