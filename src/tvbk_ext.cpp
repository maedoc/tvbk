#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include "tvbk.hpp"

namespace nb = nanobind;
using namespace nb::literals;
typedef nb::ndarray<float, nb::numpy, nb::device::cpu, nb::shape<-1>,
                    nb::c_contig>
    fvec;
typedef nb::ndarray<float, nb::numpy, nb::device::cpu, nb::shape<-1, -1>,
                    nb::c_contig>
    fmat;
typedef nb::ndarray<float, nb::numpy, nb::device::cpu, nb::shape<-1, -1, -1>,
                    nb::c_contig>
    farr3;
typedef nb::ndarray<float, nb::numpy, nb::device::cpu,
                    nb::shape<-1, -1, -1, -1>, nb::c_contig>
    farr4;
typedef nb::ndarray<uint32_t, nb::numpy, nb::device::cpu, nb::shape<-1>,
                    nb::c_contig>
    uvec;

template <typename shape>
using farr =
    nb::ndarray<float, nb::numpy, nb::device::cpu, shape, nb::c_contig>;

extern "C" void coupling_kernel_cpu(void *out, const void **in);

extern "C" void coupling_batch_kernel_cpu(void *out, const void **in);

// boilerplate for declaring a model stepping function
template <typename model, typename M, int width = 8> void decl_step(M m) {
  char name[64];
  snprintf(name, 64, "step_%s", model::name);
  m.def(
      name,
      [](const tvbk::cxbs<width> &cx, const tvbk::conn &c,
         farr<nb::shape<-1, model::num_svar, -1, width>> &x,
         farr<nb::shape<-1, model::num_svar, -1, width>> &y,
         farr<nb::shape<-1, model::num_svar, width>> &z,
         farr<nb::shape<-1, -1, model::num_parm, width>> &p, uint32_t t0,
         uint32_t nt, float dt,
         nb::ndarray<uint64_t, nb::numpy, nb::device::cpu,
                     nb::shape<-1, width, 4>>
             seed) {
        std::string err_msg = "no error";

        // Check for zero delays in connectivity
        for (uint32_t i = 0; i < c.num_nonzero; i++) {
          if (c.idelays[i] == 0) {
            err_msg = "Conn contains zero delays, which are not supported "
                      "(must be >= 1).";
            goto throw_error;
          }
        }

        // check x.shape[0] == p.shape[0] == cx.num_batch
        if (!(x.shape(0) == p.shape(0) && x.shape(0) == cx.num_batch)) {
          err_msg = "batch shapes don't match: check that x.shape[0] == "
                    "p.shape[0] == cx.num_batch";
          goto throw_error;
        }
        // check x.shape == y.shape
        if (!(x.shape(0) == y.shape(0) && x.shape(2) == y.shape(2))) {
          err_msg = "x and y shapes do not match!";
          goto throw_error;
        }
        // check z.shape
        if (!(x.shape(0) == z.shape(0) && seed.shape(0) == x.shape(0))) {
          err_msg = "z.shape or seed.shape doesn't match x";
          goto throw_error;
        }
        // check x.shape[2] == cx.num_node == p.shape[1]
        if (!(x.shape(2) == cx.num_node && x.shape(2))) {
          err_msg =
              "node shapes don't match: check that x.shape[2] == cx.num_node";
          goto throw_error;
        }
        if (!(p.shape(1) == cx.num_node || p.shape(1) == 1)) {
          err_msg = "p.shape[1] must be num_node or 1.";
          goto throw_error;
        }
        tvbk::step_batches<model, 8>(
            cx, c, (float *)x.data(), (float *)y.data(), (float *)z.data(),
            (float *)p.data(), p.shape(1) == cx.num_node, t0, nt, dt,
            (uint64_t *)seed.data());
        return;
      throw_error:
#ifdef __EMSCRIPTEN__
        char log_cmd[256];
        snprintf(log_cmd, 256, "console.error('%s')", err_msg.c_str());
        emscripten_run_script(log_cmd);
#else
        throw std::runtime_error(err_msg);
#endif
      },
      "cx"_a, "c"_a, "x"_a, "y"_a, "z"_a, "p"_a, "t0"_a, "nt"_a, "dt"_a,
      "seed"_a);
}

// exposing heun_step for fine-grained control
template <typename model, typename M, int width = 8>
void decl_step_integrate(M m) {
  char name[64];
  snprintf(name, 64, "step_integrate_%s%d", model::name, width);
  m.def(
      name,
      [](const tvbk::cxb<width> &cx,
         farr<nb::shape<model::num_svar, -1, width>> &x,
         farr<nb::shape<model::num_svar, width>> &z,
         farr<nb::shape<-1, model::num_parm, width>> &p, uint32_t t, float dt,
         nb::ndarray<uint64_t, nb::numpy, nb::device::cpu, nb::shape<width, 4>>
             seed) {
        std::string err_msg = "no error";
        bool p_varies_node = false;

        // check x.shape[1] == cx.num_node
        if (x.shape(1) != cx.num_node) {
          err_msg = "x.shape[1] must match cx.num_node";
          goto throw_error;
        }
        // check p.shape[0]
        if (!(p.shape(0) == cx.num_node || p.shape(0) == 1)) {
          err_msg = "p.shape[0] must be num_node or 1.";
          goto throw_error;
        }

        p_varies_node = (p.shape(0) == cx.num_node);

        for (uint32_t i = 0; i < cx.num_node; i++) {
          const float *pi =
              p_varies_node ? (float *)p.data() + i * model::num_parm * width
                            : (float *)p.data();

          tvbk::heun_step<model, width>(
              cx, (float *)x.data(), (float *)z.data(), cx.cx1 + i * width,
              cx.cx2 + i * width, pi, i, t, dt, (uint64_t *)seed.data());
        }
        return;
      throw_error:
#ifdef __EMSCRIPTEN__
        char log_cmd[256];
        snprintf(log_cmd, 256, "console.error('%s')", err_msg.c_str());
        emscripten_run_script(log_cmd);
#else
        throw std::runtime_error(err_msg);
#endif
      },
      "cx"_a, "x"_a, "z"_a, "p"_a, "t"_a, "dt"_a, "seed"_a);
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
      .def_prop_ro(
          "buf",
          [](tvbk::cx &cx) { return fmat(cx.buf, {cx.num_node, cx.num_time}); })
      .def_prop_ro("cx1",
                   [](tvbk::cx &cx) { return fvec(cx.cx1, {cx.num_node}); })
      .def_prop_ro("cx2",
                   [](tvbk::cx &cx) { return fvec(cx.cx2, {cx.num_node}); });

  nb::class_<tvbk::cx8>(m, "Cx8")
      .def(nb::init<uint32_t, uint32_t>(), "num_node"_a, "num_time"_a)
      .def_ro("num_node", &tvbk::cx8::num_node)
      .def_ro("num_time", &tvbk::cx8::num_time)
      .def_ro("num_item", &tvbk::cx8::num_item)
      .def_prop_ro("buf",
                   [](tvbk::cx8 &cx) {
                     return fmat(cx.buf,
                                 {cx.num_node, cx.num_time, cx.num_item});
                   })
      .def_prop_ro("cx1",
                   [](tvbk::cx8 &cx) {
                     return fmat(cx.cx1, {cx.num_node, cx.num_item});
                   })
      .def_prop_ro("cx2", [](tvbk::cx8 &cx) {
        return fmat(cx.cx2, {cx.num_node, cx.num_item});
      });

  nb::class_<tvbk::cx8s>(m, "Cx8s")
      .def(nb::init<uint32_t, uint32_t, uint32_t>(), "num_node"_a, "num_time"_a,
           "num_batches"_a)
      .def_ro("num_node", &tvbk::cx8s::num_node)
      .def_ro("num_time", &tvbk::cx8s::num_time)
      .def_ro("num_item", &tvbk::cx8s::num_item)
      .def_ro("num_batch", &tvbk::cx8s::num_batch)
      .def_prop_ro("buf",
                   [](tvbk::cx8s &cx) {
                     return farr4(cx.buf, {cx.num_batch, cx.num_node,
                                           cx.num_time, cx.num_item});
                   })
      .def_prop_ro("cx1",
                   [](tvbk::cx8s &cx) {
                     return farr3(cx.cx1,
                                  {cx.num_batch, cx.num_node, cx.num_item});
                   })
      .def_prop_ro("cx2", [](tvbk::cx8s &cx) {
        return farr3(cx.cx2, {cx.num_batch, cx.num_node, cx.num_item});
      });

  nb::class_<tvbk::conn>(m, "Conn")
      .def(nb::init<uint32_t, uint32_t>(), "num_node"_a, "num_nonzero"_a)
      .def_ro("num_node", &tvbk::conn::num_node)
      .def_ro("num_nonzero", &tvbk::conn::num_nonzero)
      .def_prop_ro("weights",
                   [](tvbk::conn &c) {
                     return fvec(const_cast<float *>(c.weights),
                                 {c.num_nonzero});
                   })
      .def_prop_ro("indices",
                   [](tvbk::conn &c) {
                     return uvec(const_cast<uint32_t *>(c.indices),
                                 {c.num_nonzero});
                   })
      .def_prop_ro("indptr",
                   [](tvbk::conn &c) {
                     return uvec(const_cast<uint32_t *>(c.indptr),
                                 {c.num_node + 1});
                   })
      .def_prop_ro("idelays", [](tvbk::conn &c) {
        return uvec(const_cast<uint32_t *>(c.idelays), {c.num_nonzero});
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

  m.def(
      "cxs8_j",
      [](const tvbk::cx8s &cxs8, const tvbk::conn &conn, uint32_t t) {
        tvbk::cx_j_bs<8>(cxs8, conn, t);
      },
      "cxs8"_a, "conn"_a, "t"_a,
      "This function calculates many batched afferent coupling buffer.");

  m.def(
      "dfun_jr8",
      [](farr<nb::shape<tvbk::jr::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::jr::num_svar, 8>> &x,
         farr<nb::shape<tvbk::jr::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::jr::num_parm, 8>> &p) {
        tvbk::jr::dfun<8>((float *)dx.data(), (float *)x.data(),
                          (float *)c.data(), (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  m.def(
      "dfun_mpr8",
      [](farr<nb::shape<tvbk::mpr::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::mpr::num_svar, 8>> &x,
         farr<nb::shape<tvbk::mpr::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::mpr::num_parm, 8>> &p) {
        tvbk::mpr::dfun<8>((float *)dx.data(), (float *)x.data(),
                           (float *)c.data(), (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  m.def(
      "dfun_kionex8",
      [](farr<nb::shape<tvbk::kionex::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::kionex::num_svar, 8>> &x,
         farr<nb::shape<tvbk::kionex::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::kionex::num_parm, 8>> &p) {
        tvbk::kionex::dfun<8>((float *)dx.data(), (float *)x.data(),
                              (float *)c.data(), (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  m.def(
      "dfun_kionex2_8",
      [](farr<nb::shape<tvbk::kionex2::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::kionex2::num_svar, 8>> &x,
         farr<nb::shape<tvbk::kionex2::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::kionex2::num_parm, 8>> &p) {
        tvbk::kionex2::dfun<8>((float *)dx.data(), (float *)x.data(),
                               (float *)c.data(), (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  decl_step<tvbk::jr>(m);
  decl_step<tvbk::mpr>(m);
  decl_step<tvbk::kionex>(m);
  decl_step<tvbk::kionex2>(m);

  decl_step_integrate<tvbk::jr>(m);
  decl_step_integrate<tvbk::mpr>(m);
  decl_step_integrate<tvbk::kionex>(m);
  decl_step_integrate<tvbk::kionex2>(m);

  // Batch 1 models
  decl_step<tvbk::kuramoto>(m);
  decl_step<tvbk::sup_hopf>(m);
  decl_step<tvbk::generic_2d>(m);
  decl_step<tvbk::wilson_cowan>(m);
  decl_step<tvbk::reduced_wong_wang>(m);

  decl_step_integrate<tvbk::kuramoto>(m);
  decl_step_integrate<tvbk::sup_hopf>(m);
  decl_step_integrate<tvbk::generic_2d>(m);
  decl_step_integrate<tvbk::wilson_cowan>(m);
  decl_step_integrate<tvbk::reduced_wong_wang>(m);

  m.def(
      "dfun_kuramoto8",
      [](farr<nb::shape<tvbk::kuramoto::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::kuramoto::num_svar, 8>> &x,
         farr<nb::shape<tvbk::kuramoto::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::kuramoto::num_parm, 8>> &p) {
        tvbk::kuramoto::dfun<8>((float *)dx.data(), (float *)x.data(),
                                (float *)c.data(), (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  m.def(
      "dfun_sup_hopf8",
      [](farr<nb::shape<tvbk::sup_hopf::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::sup_hopf::num_svar, 8>> &x,
         farr<nb::shape<tvbk::sup_hopf::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::sup_hopf::num_parm, 8>> &p) {
        tvbk::sup_hopf::dfun<8>((float *)dx.data(), (float *)x.data(),
                                (float *)c.data(), (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  m.def(
      "dfun_generic_2d8",
      [](farr<nb::shape<tvbk::generic_2d::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::generic_2d::num_svar, 8>> &x,
         farr<nb::shape<tvbk::generic_2d::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::generic_2d::num_parm, 8>> &p) {
        tvbk::generic_2d::dfun<8>((float *)dx.data(), (float *)x.data(),
                                  (float *)c.data(), (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  m.def(
      "dfun_wilson_cowan8",
      [](farr<nb::shape<tvbk::wilson_cowan::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::wilson_cowan::num_svar, 8>> &x,
         farr<nb::shape<tvbk::wilson_cowan::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::wilson_cowan::num_parm, 8>> &p) {
        tvbk::wilson_cowan::dfun<8>((float *)dx.data(), (float *)x.data(),
                                    (float *)c.data(), (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  m.def(
      "dfun_reduced_wong_wang8",
      [](farr<nb::shape<tvbk::reduced_wong_wang::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::reduced_wong_wang::num_svar, 8>> &x,
         farr<nb::shape<tvbk::reduced_wong_wang::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::reduced_wong_wang::num_parm, 8>> &p) {
        tvbk::reduced_wong_wang::dfun<8>((float *)dx.data(), (float *)x.data(),
                                         (float *)c.data(), (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  decl_step<tvbk::epileptor>(m);
  decl_step_integrate<tvbk::epileptor>(m);
  m.def(
      "dfun_epileptor8",
      [](farr<nb::shape<tvbk::epileptor::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::epileptor::num_svar, 8>> &x,
         farr<nb::shape<tvbk::epileptor::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::epileptor::num_parm, 8>> &p) {
        tvbk::epileptor::dfun<8>((float *)dx.data(), (float *)x.data(),
                                 (float *)c.data(), (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  decl_step<tvbk::epileptor_2d>(m);
  decl_step_integrate<tvbk::epileptor_2d>(m);
  m.def(
      "dfun_epileptor_2d8",
      [](farr<nb::shape<tvbk::epileptor_2d::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::epileptor_2d::num_svar, 8>> &x,
         farr<nb::shape<tvbk::epileptor_2d::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::epileptor_2d::num_parm, 8>> &p) {
        tvbk::epileptor_2d::dfun<8>((float *)dx.data(), (float *)x.data(),
                                    (float *)c.data(), (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  decl_step<tvbk::linear>(m);
  decl_step_integrate<tvbk::linear>(m);
  m.def(
      "dfun_linear8",
      [](farr<nb::shape<tvbk::linear::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::linear::num_svar, 8>> &x,
         farr<nb::shape<tvbk::linear::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::linear::num_parm, 8>> &p) {
        tvbk::linear::dfun<8>((float *)dx.data(), (float *)x.data(),
                              (float *)c.data(), (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  decl_step<tvbk::hopfield>(m);
  decl_step_integrate<tvbk::hopfield>(m);
  m.def(
      "dfun_hopfield8",
      [](farr<nb::shape<tvbk::hopfield::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::hopfield::num_svar, 8>> &x,
         farr<nb::shape<tvbk::hopfield::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::hopfield::num_parm, 8>> &p) {
        tvbk::hopfield::dfun<8>((float *)dx.data(), (float *)x.data(),
                                (float *)c.data(), (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  decl_step<tvbk::hopfield_dynamic>(m);
  decl_step_integrate<tvbk::hopfield_dynamic>(m);
  m.def(
      "dfun_hopfield_dynamic8",
      [](farr<nb::shape<tvbk::hopfield_dynamic::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::hopfield_dynamic::num_svar, 8>> &x,
         farr<nb::shape<tvbk::hopfield_dynamic::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::hopfield_dynamic::num_parm, 8>> &p) {
        tvbk::hopfield_dynamic::dfun<8>((float *)dx.data(), (float *)x.data(),
                                        (float *)c.data(), (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  decl_step<tvbk::larter_breakspear>(m);
  decl_step_integrate<tvbk::larter_breakspear>(m);
  m.def(
      "dfun_larter_breakspear8",
      [](farr<nb::shape<tvbk::larter_breakspear::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::larter_breakspear::num_svar, 8>> &x,
         farr<nb::shape<tvbk::larter_breakspear::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::larter_breakspear::num_parm, 8>> &p) {
        tvbk::larter_breakspear::dfun<8>((float *)dx.data(), (float *)x.data(),
                                         (float *)c.data(), (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  decl_step<tvbk::infinite_theta>(m);
  decl_step_integrate<tvbk::infinite_theta>(m);
  m.def(
      "dfun_infinite_theta8",
      [](farr<nb::shape<tvbk::infinite_theta::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::infinite_theta::num_svar, 8>> &x,
         farr<nb::shape<tvbk::infinite_theta::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::infinite_theta::num_parm, 8>> &p) {
        tvbk::infinite_theta::dfun<8>((float *)dx.data(), (float *)x.data(),
                                      (float *)c.data(), (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  decl_step<tvbk::epileptor_rs>(m);
  decl_step_integrate<tvbk::epileptor_rs>(m);
  m.def(
      "dfun_epileptor_rs8",
      [](farr<nb::shape<tvbk::epileptor_rs::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::epileptor_rs::num_svar, 8>> &x,
         farr<nb::shape<tvbk::epileptor_rs::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::epileptor_rs::num_parm, 8>> &p) {
        tvbk::epileptor_rs::dfun<8>((float *)dx.data(), (float *)x.data(),
                                    (float *)c.data(), (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  decl_step<tvbk::epileptor_codim3>(m);
  decl_step_integrate<tvbk::epileptor_codim3>(m);
  m.def(
      "dfun_epileptor_codim3_8",
      [](farr<nb::shape<tvbk::epileptor_codim3::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::epileptor_codim3::num_svar, 8>> &x,
         farr<nb::shape<tvbk::epileptor_codim3::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::epileptor_codim3::num_parm, 8>> &p) {
        tvbk::epileptor_codim3::dfun<8>((float *)dx.data(), (float *)x.data(),
                                        (float *)c.data(), (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  decl_step<tvbk::epileptor_codim3_slow_mod>(m);
  decl_step_integrate<tvbk::epileptor_codim3_slow_mod>(m);
  m.def(
      "dfun_epileptor_codim3_slow_mod_8",
      [](farr<nb::shape<tvbk::epileptor_codim3_slow_mod::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::epileptor_codim3_slow_mod::num_svar, 8>> &x,
         farr<nb::shape<tvbk::epileptor_codim3_slow_mod::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::epileptor_codim3_slow_mod::num_parm, 8>> &p) {
        tvbk::epileptor_codim3_slow_mod::dfun<8>(
            (float *)dx.data(), (float *)x.data(), (float *)c.data(),
            (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  decl_step<tvbk::coombes_byrne>(m);
  decl_step_integrate<tvbk::coombes_byrne>(m);
  m.def(
      "dfun_coombes_byrne_8",
      [](farr<nb::shape<tvbk::coombes_byrne::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::coombes_byrne::num_svar, 8>> &x,
         farr<nb::shape<tvbk::coombes_byrne::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::coombes_byrne::num_parm, 8>> &p) {
        tvbk::coombes_byrne::dfun<8>((float *)dx.data(), (float *)x.data(),
                                     (float *)c.data(), (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  decl_step<tvbk::coombes_byrne_2d>(m);
  decl_step_integrate<tvbk::coombes_byrne_2d>(m);
  m.def(
      "dfun_coombes_byrne_2d_8",
      [](farr<nb::shape<tvbk::coombes_byrne_2d::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::coombes_byrne_2d::num_svar, 8>> &x,
         farr<nb::shape<tvbk::coombes_byrne_2d::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::coombes_byrne_2d::num_parm, 8>> &p) {
        tvbk::coombes_byrne_2d::dfun<8>((float *)dx.data(), (float *)x.data(),
                                        (float *)c.data(), (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  decl_step<tvbk::gast_schmidt_knosche_sd>(m);
  decl_step_integrate<tvbk::gast_schmidt_knosche_sd>(m);
  m.def(
      "dfun_gast_schmidt_knosche_sd_8",
      [](farr<nb::shape<tvbk::gast_schmidt_knosche_sd::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::gast_schmidt_knosche_sd::num_svar, 8>> &x,
         farr<nb::shape<tvbk::gast_schmidt_knosche_sd::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::gast_schmidt_knosche_sd::num_parm, 8>> &p) {
        tvbk::gast_schmidt_knosche_sd::dfun<8>(
            (float *)dx.data(), (float *)x.data(), (float *)c.data(),
            (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  decl_step<tvbk::gast_schmidt_knosche_sf>(m);
  decl_step_integrate<tvbk::gast_schmidt_knosche_sf>(m);
  m.def(
      "dfun_gast_schmidt_knosche_sf_8",
      [](farr<nb::shape<tvbk::gast_schmidt_knosche_sf::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::gast_schmidt_knosche_sf::num_svar, 8>> &x,
         farr<nb::shape<tvbk::gast_schmidt_knosche_sf::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::gast_schmidt_knosche_sf::num_parm, 8>> &p) {
        tvbk::gast_schmidt_knosche_sf::dfun<8>(
            (float *)dx.data(), (float *)x.data(), (float *)c.data(),
            (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  decl_step<tvbk::zetterberg_jansen>(m);
  decl_step_integrate<tvbk::zetterberg_jansen>(m);
  m.def(
      "dfun_zetterberg_jansen_8",
      [](farr<nb::shape<tvbk::zetterberg_jansen::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::zetterberg_jansen::num_svar, 8>> &x,
         farr<nb::shape<tvbk::zetterberg_jansen::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::zetterberg_jansen::num_parm, 8>> &p) {
        tvbk::zetterberg_jansen::dfun<8>((float *)dx.data(), (float *)x.data(),
                                         (float *)c.data(), (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  decl_step<tvbk::zerlaut_adaptation_first_order>(m);
  decl_step_integrate<tvbk::zerlaut_adaptation_first_order>(m);
  m.def(
      "dfun_zerlaut_adaptation_first_order_8",
      [](farr<nb::shape<tvbk::zerlaut_adaptation_first_order::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::zerlaut_adaptation_first_order::num_svar, 8>> &x,
         farr<nb::shape<tvbk::zerlaut_adaptation_first_order::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::zerlaut_adaptation_first_order::num_parm, 8>>
             &p) {
        tvbk::zerlaut_adaptation_first_order::dfun<8>(
            (float *)dx.data(), (float *)x.data(), (float *)c.data(),
            (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  decl_step<tvbk::zerlaut_adaptation_second_order>(m);
  decl_step_integrate<tvbk::zerlaut_adaptation_second_order>(m);
  m.def(
      "dfun_zerlaut_adaptation_second_order_8",
      [](farr<nb::shape<tvbk::zerlaut_adaptation_second_order::num_svar, 8>>
             &dx,
         farr<nb::shape<tvbk::zerlaut_adaptation_second_order::num_svar, 8>> &x,
         farr<nb::shape<tvbk::zerlaut_adaptation_second_order::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::zerlaut_adaptation_second_order::num_parm, 8>>
             &p) {
        tvbk::zerlaut_adaptation_second_order::dfun<8>(
            (float *)dx.data(), (float *)x.data(), (float *)c.data(),
            (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  decl_step<tvbk::reduced_wong_wang_exc_inh>(m);
  decl_step_integrate<tvbk::reduced_wong_wang_exc_inh>(m);
  m.def(
      "dfun_reduced_wong_wang_exc_inh_8",
      [](farr<nb::shape<tvbk::reduced_wong_wang_exc_inh::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::reduced_wong_wang_exc_inh::num_svar, 8>> &x,
         farr<nb::shape<tvbk::reduced_wong_wang_exc_inh::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::reduced_wong_wang_exc_inh::num_parm, 8>> &p) {
        tvbk::reduced_wong_wang_exc_inh::dfun<8>(
            (float *)dx.data(), (float *)x.data(), (float *)c.data(),
            (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  decl_step<tvbk::deco_balanced_exc_inh>(m);
  decl_step_integrate<tvbk::deco_balanced_exc_inh>(m);
  m.def(
      "dfun_deco_balanced_exc_inh_8",
      [](farr<nb::shape<tvbk::deco_balanced_exc_inh::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::deco_balanced_exc_inh::num_svar, 8>> &x,
         farr<nb::shape<tvbk::deco_balanced_exc_inh::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::deco_balanced_exc_inh::num_parm, 8>> &p) {
        tvbk::deco_balanced_exc_inh::dfun<8>(
            (float *)dx.data(), (float *)x.data(), (float *)c.data(),
            (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  decl_step<tvbk::reduced_set_fitz_hugh_nagumo>(m);
  decl_step_integrate<tvbk::reduced_set_fitz_hugh_nagumo>(m);
  m.def(
      "dfun_reduced_set_fitz_hugh_nagumo_8",
      [](farr<nb::shape<tvbk::reduced_set_fitz_hugh_nagumo::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::reduced_set_fitz_hugh_nagumo::num_svar, 8>> &x,
         farr<nb::shape<tvbk::reduced_set_fitz_hugh_nagumo::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::reduced_set_fitz_hugh_nagumo::num_parm, 8>> &p) {
        tvbk::reduced_set_fitz_hugh_nagumo::dfun<8>(
            (float *)dx.data(), (float *)x.data(), (float *)c.data(),
            (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  decl_step<tvbk::reduced_set_hindmarsh_rose>(m);
  decl_step_integrate<tvbk::reduced_set_hindmarsh_rose>(m);
  m.def(
      "dfun_reduced_set_hindmarsh_rose_8",
      [](farr<nb::shape<tvbk::reduced_set_hindmarsh_rose::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::reduced_set_hindmarsh_rose::num_svar, 8>> &x,
         farr<nb::shape<tvbk::reduced_set_hindmarsh_rose::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::reduced_set_hindmarsh_rose::num_parm, 8>> &p) {
        tvbk::reduced_set_hindmarsh_rose::dfun<8>(
            (float *)dx.data(), (float *)x.data(), (float *)c.data(),
            (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);

  decl_step<tvbk::dumont_gutkin>(m);
  decl_step_integrate<tvbk::dumont_gutkin>(m);
  m.def(
      "dfun_dumont_gutkin_8",
      [](farr<nb::shape<tvbk::dumont_gutkin::num_svar, 8>> &dx,
         farr<nb::shape<tvbk::dumont_gutkin::num_svar, 8>> &x,
         farr<nb::shape<tvbk::dumont_gutkin::num_cvar, 8>> &c,
         farr<nb::shape<tvbk::dumont_gutkin::num_parm, 8>> &p) {
        tvbk::dumont_gutkin::dfun<8>((float *)dx.data(), (float *)x.data(),
                                     (float *)c.data(), (float *)p.data());
      },
      "dx"_a, "x"_a, "c"_a, "p"_a);
  m.def("registrations", []() {
    nb::dict dict;
    dict["coupling_kernel_cpu"] =
        nb::capsule((void *)coupling_kernel_cpu, "xla._CUSTOM_CALL_TARGET");
    dict["coupling_batch_kernel_cpu"] = nb::capsule(
        (void *)coupling_batch_kernel_cpu, "xla._CUSTOM_CALL_TARGET");
    return dict;
  });
}
