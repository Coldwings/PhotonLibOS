/*
Copyright 2022 The Photon Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "iouring-wrapper.h"

#include <unistd.h>
#include <sys/eventfd.h>
#include <sys/resource.h>
#include <fcntl.h>
#include <cstdint>
#include <limits>
#include <atomic>
#include <unordered_map>

#include <liburing.h>
#include <photon/common/alog.h>
#include <photon/thread/thread11.h>
#include <photon/io/fd-events.h>
#include "events_map.h"
#include "reset_handle.h"

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#define container_of(ptr, type, member) \
    ((type*)((char*)static_cast<const decltype(((type*)0)->member)*>(ptr) - offsetof(type,member)))

namespace photon {

constexpr static EventsMap<EVUnderlay<POLLIN | POLLRDHUP, POLLOUT, POLLERR>> evmap;

class iouringEngine : public MasterEventEngine, public CascadingEventEngine, public ResetHandle {
public:
    ~iouringEngine() {
        LOG_INFO("Finish event engine: iouring ",
            make_named_value("is_master", m_args.is_master));
        fini();
    }

    int reset() override {
        fini();
        m_event_contexts.clear();
        return init();
    }

    int fini() {
        if (m_eventfd >= 0) {
            if (!m_args.is_master) {
                if (io_uring_unregister_eventfd(m_ring) != 0)
                    LOG_ERROR("iouring: failed to unregister cascading event fd ", ERRNO());
            }
            close(m_eventfd);
        }
        if (m_ring != nullptr) {
            io_uring_queue_exit(m_ring);
        }
        delete m_ring;
        m_ring = nullptr;
        return 0;
    }

    int init() { return init(m_args); }
    int init(iouring_args args) {
        m_args = args;
        m_args.setup_sq_aff = false;
        int compare_result;
        if (kernel_version_compare("5.11", compare_result) == 0 && compare_result <= 0) {
            rlimit resource_limit{.rlim_cur = RLIM_INFINITY, .rlim_max = RLIM_INFINITY};
            if (setrlimit(RLIMIT_MEMLOCK, &resource_limit) != 0)
                LOG_ERROR_RETURN(0, -1, "iouring: failed to set resource limit. "
                                        "Use command `ulimit -l unlimited`, or change to root");
        }

        check_register_file_support();
        check_cooperative_task_support();
        set_submit_wait_function();

        m_ring = new io_uring{};
        io_uring_params params{};
        if (m_cooperative_task_flag == 1) {
            params.flags = IORING_SETUP_COOP_TASKRUN;
        }
        if (args.setup_iopoll)
            params.flags |= IORING_SETUP_IOPOLL;
        if (args.setup_sqpoll) {
            params.flags |= IORING_SETUP_SQPOLL;
            params.sq_thread_idle = args.sq_thread_idle_ms;
            if (args.setup_sq_aff) {
                params.flags |= IORING_SETUP_SQ_AFF;
                params.sq_thread_cpu = args.sq_thread_cpu;
            }
        }

    retry:
        int ret = io_uring_queue_init_params(QUEUE_DEPTH, m_ring, &params);
        if (ret != 0) {
            if (-ret == EINVAL) {
                auto& p = params;
                if (p.flags & IORING_SETUP_DEFER_TASKRUN) {
                    p.flags &= ~IORING_SETUP_DEFER_TASKRUN;
                    p.flags &= ~IORING_SETUP_SINGLE_ISSUER;
                    LOG_INFO("io_uring_queue_init failed, removing IORING_SETUP_DEFER_TASKRUN, IORING_SETUP_SINGLE_ISSUER");
                    goto retry;
                }
                if (p.flags & IORING_SETUP_COOP_TASKRUN) {
                    // this seems to be conflicting with IORING_SETUP_SQPOLL,
                    // at least in 6.4.12-1.el8.elrepo.x86_64
                    p.flags &= ~IORING_SETUP_COOP_TASKRUN;
                    LOG_INFO("io_uring_queue_init failed, removing IORING_SETUP_COOP_TASKRUN");
                    goto retry;
                }
                if (p.flags & IORING_SETUP_CQSIZE) {
                    p.flags &= ~IORING_SETUP_CQSIZE;
                    LOG_INFO("io_uring_queue_init failed, removing IORING_SETUP_CQSIZE");
                    goto retry;
            }   }
            // reset m_ring so that the destructor won't do duplicate munmap cleanup (io_uring_queue_exit)
            delete m_ring;
            m_ring = nullptr;
            LOG_ERROR_RETURN(0, -1, "iouring: failed to init queue: ", ERRNO(-ret));
        }

        // Check feature supported
        if (!check_required_features(params, IORING_FEAT_CUR_PERSONALITY,
                        IORING_FEAT_NODROP,  IORING_FEAT_FAST_POLL,
                        IORING_FEAT_EXT_ARG, IORING_FEAT_RW_CUR_POS)) {
            LOG_ERROR_RETURN(0, -1, "iouring: required feature not supported");
        }

        // Check opcode supported
        auto probe = io_uring_get_probe_ring(m_ring);
        if (probe == nullptr) {
            LOG_ERROR_RETURN(0, -1, "iouring: failed to get probe");
        }
        DEFER(io_uring_free_probe(probe));
        if (!io_uring_opcode_supported(probe, IORING_OP_PROVIDE_BUFFERS) ||
            !io_uring_opcode_supported(probe, IORING_OP_ASYNC_CANCEL)) {
            LOG_ERROR_RETURN(0, -1, "iouring: some opcodes are not supported");
        }

        // Register ring fd, only available since 5.18. Doesn't have to succeed
        ret = io_uring_register_ring_fd(m_ring);
        if (ret < 0 && ret != -EINVAL) {
            LOG_ERROR_RETURN(EINVAL, -1, "iouring: unable to register ring fd", ret);
        }

        // Register files. Init with sparse entries
        if (register_files_enabled()) {
            auto entries = new int[REGISTER_FILES_MAX_NUM];
            DEFER(delete[] entries);
            for (int i = 0; i < REGISTER_FILES_MAX_NUM; ++i) {
                entries[i] = REGISTER_FILES_SPARSE_FD;
            }
            ret = io_uring_register_files(m_ring, entries, REGISTER_FILES_MAX_NUM);
            if (ret != 0) {
                LOG_ERROR_RETURN(-ret, -1, "iouring: unable to register files, ", ERRNO(-ret));
            }
        }

        m_eventfd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
        if (m_eventfd < 0) {
            LOG_ERRNO_RETURN(0, -1, "iouring: failed to create eventfd");
        }

        if (args.is_master) {
            // Setup a multishot poll on master engine to watch the cancel_wait
            uint32_t poll_mask = evmap.translate_bitwisely(EVENT_READ);
            auto sqe = _get_sqe();
            if (!sqe) return -1;
            io_uring_prep_poll_multishot(sqe, m_eventfd, poll_mask);
            io_uring_sqe_set_data(sqe, this);
            ret = io_uring_submit(m_ring);
            if (ret <= 0) {
                LOG_ERROR_RETURN(-ret, -1, "iouring: fail to submit multishot poll, ", ERRNO(-ret));
            }
        } else {
            // Register cascading engine to eventfd
            if (io_uring_register_eventfd(m_ring, m_eventfd) != 0) {
                LOG_ERRNO_RETURN(0, -1, "iouring: failed to register cascading event fd");
            }
        }
        return 0;
    }


    template<typename T, typename...Ts>
    bool check_required_features(const io_uring_params& params, T f, Ts...fs) {
        return (params.features & f) && check_required_features(params, fs...);
    }
    bool check_required_features(const io_uring_params& params) {
        return true;
    }

    /**
     * @brief Get a SQE from ring, prepare IO, and wait for completion. Note all the SQEs are batch submitted
     *     later in the `wait_and_fire_events`.
     * @param timeout Timeout in usec. It could be 0 (immediate cancel), and -1 (most efficient way, no linked SQE).
     *     Note that the cancelling has no guarantee to succeed, it's just an attempt.
     * @param ring_flags The lowest 8 bits is for sqe.flags. The rest is reserved.
     * @retval Non negative integers for success, -1 for failure. If failed with timeout, errno will
     *     be set to ETIMEDOUT. If failed because of external interruption, errno will also be set accordingly.
     */
    template<typename Prep, typename... Args>
    int32_t async_io(Prep prep, Timeout timeout, uint32_t ring_flags, Args... args) {
        auto* sqe = _get_sqe();
        if (sqe == nullptr)
            return -1;
        prep(sqe, args...);
        return _async_io(sqe, timeout, ring_flags);
    }

    int32_t _async_io(io_uring_sqe* sqe, Timeout timeout, uint32_t ring_flags) {
        sqe->flags |= (uint8_t) (ring_flags & 0xff);
        ioCtx io_ctx(false, false);
        io_uring_sqe_set_data(sqe, &io_ctx);

        ioCtx timer_ctx(true, false);
        __kernel_timespec ts;
        auto usec = timeout.timeout_us();
        if (usec < (uint64_t) std::numeric_limits<int64_t>::max()) {
            sqe->flags |= IOSQE_IO_LINK;
            ts = usec_to_timespec(usec);
            sqe = _get_sqe();
            if (sqe == nullptr)
                return -1;
            io_uring_prep_link_timeout(sqe, &ts, 0);
            io_uring_sqe_set_data(sqe, &timer_ctx);
        }

        if (try_submit() < 0) return -1;

        SCOPED_PAUSE_WORK_STEALING;
        photon::thread_sleep(-1);

        if (likely(errno == EOK)) {
            // Interrupted by `wait_and_fire_events`
            if (io_ctx.res < 0) {
                errno = -io_ctx.res;
                return -1;
            } else {
                // errno = 0;
                return io_ctx.res;
            }
        } else {
            // Interrupted by external user thread. Try to cancel the previous I/O
            ERRNO err_backup;
            sqe = _get_sqe();
            if (sqe == nullptr)
                return -1;
            ioCtx cancel_ctx(true, false);
            io_uring_prep_cancel(sqe, &io_ctx, 0);
            io_uring_sqe_set_data(sqe, &cancel_ctx);
            photon::thread_sleep(-1);
            errno = err_backup.no;
            return -1;
        }
    }

    int try_submit() {
        if (m_args.eager_submit) {
            int ret = io_uring_submit(m_ring);
            if (ret < 0)
                LOG_ERROR_RETURN(-ret, -1, "iouring: failed to io_uring_submit(), ", ERRNO(-ret));
        }
        return 0;
    }

    int wait_for_fd(int fd, uint32_t interests, Timeout timeout) override {
        if (unlikely(interests == 0))
            return 0;
        unsigned poll_mask = evmap.translate_bitwisely(interests);
        // The io_uring_prep_poll_add's return value is the same as poll(2)'s revents.
        int ret = async_io(&io_uring_prep_poll_add, timeout, 0, fd, poll_mask);
        if (ret < 0) {
            return -1;
        }
        if (ret & POLLNVAL) {
            LOG_ERRNO_RETURN(EINVAL, -1, "iouring: poll invalid argument");
        }
        return 0;
    }
    int add_interest(Event e) override {
        auto* sqe = _get_sqe();
        if (sqe == nullptr)
            return -1;

        bool one_shot = e.interests & ONE_SHOT;
        fdInterest fd_interest{e.fd, (uint32_t)evmap.translate_bitwisely(e.interests)};
        eventCtx event_ctx{e, one_shot, {false, true}};
        auto pair = m_event_contexts.emplace(fd_interest, event_ctx);
        if (!pair.second) {
            LOG_ERROR_RETURN(0, -1, "iouring: event has already been added");
        }

        if (one_shot) {
            io_uring_prep_poll_add(sqe, fd_interest.fd, fd_interest.interest);
        } else {
            io_uring_prep_poll_multishot(sqe, fd_interest.fd, fd_interest.interest);
        }
        io_uring_sqe_set_data(sqe, &pair.first->second.io_ctx);
        return try_submit();
    }

    int rm_interest(Event e) override {
        auto* sqe = _get_sqe();
        if (sqe == nullptr)
            return -1;

        fdInterest fd_interest{e.fd, (uint32_t)evmap.translate_bitwisely(e.interests)};
        auto iter = m_event_contexts.find(fd_interest);
        if (iter == m_event_contexts.end()) {
            LOG_ERROR_RETURN(0, -1, "iouring: event is either non-existent or one-shot finished");
        }

        io_uring_prep_poll_remove(sqe, (__u64) &iter->second.io_ctx);
        io_uring_sqe_set_data(sqe, nullptr);
        return try_submit();
    }

    ssize_t wait_for_events(void** data, size_t count, Timeout timeout) override {
        if (!(m_ring->flags & IORING_SETUP_SQPOLL) &&
            m_ring->sq.sqe_tail != *m_ring->sq.khead)
                io_uring_submit(m_ring);

        // Use master engine to wait for self event fd
        int ret = ::photon::wait_for_fd_readable(m_eventfd, timeout);
        if (ret < 0) {
            return errno == ETIMEDOUT ? 0 : -1;
        }
        uint64_t value;
        eventfd_read(m_eventfd, &value);
        return reap_events(data, count);
    }
    ssize_t reap_events(void** data = 0, size_t count = 0) {
        io_uring_cqe* cqe;
        uint32_t head = 0, num = 0, n = 0;
        DEFER( io_uring_cq_advance(m_ring, n) );
        io_uring_for_each_cqe(m_ring, head, cqe) {
            n++;
            auto ctx = (ioCtx*) io_uring_cqe_get_data(cqe);
            if (!ctx) {
                // Own timeout doesn't have user data
                continue;
            }

            if (ctx == (ioCtx*) this) {
                // Triggered by cancel_wait
                eventfd_t val;
                eventfd_read(m_eventfd, &val);
                continue;
            }

            if (cqe->flags & IORING_CQE_F_NOTIF) {
                // The cqe for notify, corresponding to IORING_CQE_F_MORE
                if (unlikely(cqe->res != 0))
                    LOG_WARN("iouring: send_zc fall back to copying");
                assert(!ctx->is_event);
                photon::thread_interrupt(ctx->th_id, EOK);
                continue;
            }

            ctx->res = cqe->res;
            if (cqe->flags & IORING_CQE_F_MORE) {
                if (cqe->res & POLLERR) {
                    assert(ctx->is_event);
                    assert(num == 0);
                    LOG_ERROR_RETURN(0, -1, "iouring: multi-shot poll got POLLERR");
                } else { continue; }
            }
            if (!ctx->is_event) {
                if (!ctx->is_canceller && ctx->res == -ECANCELED) {
                    // An I/O was canceled because of:
                    // 1. IORING_OP_LINK_TIMEOUT. Leave the interrupt job to the linked timer later.
                    // 2. IORING_OP_POLL_REMOVE. The I/O is actually a polling.
                    // 3. IORING_OP_ASYNC_CANCEL. This OP is the superset of case 2.
                    ctx->res = -ETIMEDOUT;
                    continue;
                } else if (ctx->is_canceller && ctx->res == -ECANCELED) {
                    // The linked timer itself is also a canceller. The reasons it got cancelled could be:
                    // 1. I/O finished in time
                    // 2. I/O was cancelled by IORING_OP_ASYNC_CANCEL
                    continue;
                }
                photon::thread_interrupt(ctx->th_id, EOK);

            } else /* if (ctx->is_event) */ {

                if (num >= count) { --n; break; }
                eventCtx* event_ctx = container_of(ctx, eventCtx, io_ctx);
                fdInterest fd_interest{event_ctx->event.fd, (uint32_t)evmap.translate_bitwisely(event_ctx->event.interests)};
                if (ctx->res == -ECANCELED) {
                    m_event_contexts.erase(fd_interest);
                } else if (event_ctx->one_shot) {
                    data[num++] = event_ctx->event.data;
                    m_event_contexts.erase(fd_interest);
                } else {
                    data[num++] = event_ctx->event.data;
                }
            }
        }
        return num;
    }

    ssize_t wait_and_fire_events(uint64_t timeout) override {
        // Prepare own timeout
        if (timeout > (uint64_t) std::numeric_limits<int64_t>::max()) {
            timeout = std::numeric_limits<int64_t>::max();
        }

        auto ts = usec_to_timespec(timeout);
        if (m_submit_wait_func(m_ring, &ts) != 0) {
            return -1;
        }

        reap_events();
        return 0;
    }

    int cancel_wait() override {
        if (eventfd_write(m_eventfd, 1) != 0) {
            LOG_ERRNO_RETURN(0, -1, "iouring: write eventfd failed");
        }
        return 0;
    }

    static bool register_files_enabled() {
        return m_register_files_flag == 1;
    }

    int register_unregister_files(int fd, bool direction) {
        if (unlikely(!register_files_enabled())) {
            LOG_ERROR_RETURN(EINVAL, -1, "iouring: register_files not enabled");
        }
        if (unlikely(fd < 0)) {
            LOG_ERROR_RETURN(EINVAL, -1, "iouring: invalid fd to register/unregister");
        }
        if (unlikely(fd >= REGISTER_FILES_MAX_NUM)) {
            LOG_WARN("iouring: register fd exceeds max num, ignore ...");
            return 0;
        }
        if (direction) LOG_DEBUG("register", VALUE(fd)); else LOG_DEBUG("unregister", VALUE(fd));
        int value = direction ? fd : REGISTER_FILES_SPARSE_FD;
        int ret = io_uring_register_files_update(m_ring, fd, &value, 1);
        if (ret < 0) {
            LOG_ERROR_RETURN(-ret, -1, "iouring: register_files_update failed, ", VALUE(fd), ERRNO(-ret));
        }
        return 0;
    }

private:
    struct ioCtx {
        ioCtx(bool canceller, bool event) : is_canceller(canceller), is_event(event) {}
        photon::thread* th_id = photon::CURRENT;
        int32_t res = -1;
        bool is_canceller;
        bool is_event;
    };

    struct eventCtx {
        Event event;
        bool one_shot;
        ioCtx io_ctx;
    };

    struct fdInterest {
        int fd;
        uint32_t interest;

        bool operator==(const fdInterest& fi) const {
            return fi.fd == fd && fi.interest == interest;
        }
    };

    struct fdInterestHasher {
        size_t operator()(const fdInterest& key) const {
            auto ptr = (uint64_t*) &key;
            return std::hash<uint64_t>()(*ptr);
        }
    };

    io_uring_sqe* _get_sqe() {
        io_uring_sqe* sqe = io_uring_get_sqe(m_ring);
        if (sqe == nullptr) {
            LOG_ERROR_RETURN(EBUSY, sqe, "iouring: submission queue is full");
        }
        return sqe;
    }

    static int submit_wait_by_timer(io_uring* ring, __kernel_timespec* ts) {
        io_uring_sqe* sqe = io_uring_get_sqe(ring);
        if (!sqe) {
            LOG_ERROR_RETURN(EBUSY, -1, "iouring: submission queue is full");
        }
        io_uring_prep_timeout(sqe, ts, 1, 0);
        io_uring_sqe_set_data(sqe, nullptr);

        // Batch submit all SQEs
        int ret = io_uring_submit_and_wait(ring, 1);
        if (ret <= 0) {
            LOG_ERRNO_RETURN(0, -1, "iouring: failed to submit io")
        }
        return 0;
    }

    static int submit_wait_by_api(io_uring* ring, __kernel_timespec* ts) {
        // Batch submit all SQEs
        io_uring_cqe* cqe;
        int ret = io_uring_submit_and_wait_timeout(ring, &cqe, 1, ts, nullptr);
        if (ret < 0 && ret != -ETIME) {
            LOG_ERRNO_RETURN(0, -1, "iouring: failed to submit io");
        }
        return 0;
    }

    using SubmitWaitFunc = int (*)(io_uring* ring, __kernel_timespec* ts);
    static SubmitWaitFunc m_submit_wait_func;

    static void set_submit_wait_function() {
        // The submit_and_wait_timeout API is more efficient than setting up a timer and waiting for it.
        // But there is a kernel bug before 5.15, so choose appropriate function here.
        // See https://git.kernel.dk/cgit/linux-block/commit/?h=io_uring-5.17&id=228339662b398a59b3560cd571deb8b25b253c7e
        // and https://www.spinics.net/lists/stable/msg620268.html
        if (m_submit_wait_func)
            return;
        int result;
        if (kernel_version_compare("5.15", result) == 0 && result >= 0) {
            m_submit_wait_func = submit_wait_by_api;
        } else {
            m_submit_wait_func = submit_wait_by_timer;
        }
    }

    static void check_register_file_support() {
        if (m_register_files_flag >= 0)
            return;
        int result;
        if (kernel_version_compare("5.5", result) == 0 && result >= 0) {
            m_register_files_flag = 1;
            LOG_INFO("iouring: register_files is enabled");
        } else {
            m_register_files_flag = 0;
        }
    }

    static void check_cooperative_task_support() {
        if (m_cooperative_task_flag >= 0)
            return;
        int result;
        if (kernel_version_compare("5.19", result) == 0 && result >= 0) {
            m_cooperative_task_flag = 1;
        } else {
            m_cooperative_task_flag = 0;
        }
    }

    __kernel_timespec usec_to_timespec(int64_t usec) {
        int64_t sec = usec / 1000000L;
        long long nsec = (usec % 1000000L) * 1000L;
        return {sec, nsec};
    }

    static const int QUEUE_DEPTH = 16384;
    static const int REGISTER_FILES_SPARSE_FD = -1;
    static const int REGISTER_FILES_MAX_NUM = 10000;
    iouring_args m_args;
    io_uring* m_ring = nullptr;
    int m_eventfd = -1;
    std::unordered_map<fdInterest, eventCtx, fdInterestHasher> m_event_contexts;
    static int m_register_files_flag;
    static int m_cooperative_task_flag;
};

int iouringEngine::m_register_files_flag = -1;

int iouringEngine::m_cooperative_task_flag = -1;

iouringEngine::SubmitWaitFunc iouringEngine::m_submit_wait_func = nullptr;

inline iouringEngine* get_ring(CascadingEventEngine* cee) {
    return cee ? static_cast<iouringEngine*>(cee) :
                 static_cast<iouringEngine*>(get_vcpu()->master_event_engine);
}

ssize_t iouring_splice(int fd_in, int64_t off_in, int fd_out, int64_t off_out, unsigned int nbytes, uint64_t flags, Timeout timeout, CascadingEventEngine *cee) {
    uint32_t splice_flags = flags & 0xffffffff;
    uint32_t ring_flags = flags >> 32;
    return get_ring(cee)->async_io(&io_uring_prep_splice, timeout, ring_flags, fd_in, off_in, fd_out, off_out, nbytes, splice_flags);
}

ssize_t iouring_pread(int fd, void* buf, size_t count, off_t offset, uint64_t flags, Timeout timeout, CascadingEventEngine* cee) {
    uint32_t ring_flags = flags >> 32;
    return get_ring(cee)->async_io(&io_uring_prep_read, timeout, ring_flags, fd, buf, count, offset);
}

ssize_t iouring_pwrite(int fd, const void* buf, size_t count, off_t offset, uint64_t flags, Timeout timeout, CascadingEventEngine* cee) {
    uint32_t ring_flags = flags >> 32;
    return get_ring(cee)->async_io(&io_uring_prep_write, timeout, ring_flags, fd, buf, count, offset);
}

ssize_t iouring_preadv(int fd, const iovec* iov, int iovcnt, off_t offset, uint64_t flags, Timeout timeout, CascadingEventEngine* cee) {
    uint32_t ring_flags = flags >> 32;
    return get_ring(cee)->async_io(&io_uring_prep_readv, timeout, ring_flags, fd, iov, iovcnt, offset);
}

ssize_t iouring_pwritev(int fd, const iovec* iov, int iovcnt, off_t offset, uint64_t flags, Timeout timeout, CascadingEventEngine* cee) {
    uint32_t ring_flags = flags >> 32;
    return get_ring(cee)->async_io(&io_uring_prep_writev, timeout, ring_flags, fd, iov, iovcnt, offset);
}

ssize_t iouring_send(int fd, const void* buf, size_t len, uint64_t flags, Timeout timeout, CascadingEventEngine* cee) {
    uint32_t io_flags = flags & 0xffffffff;
    uint32_t ring_flags = flags >> 32;
    return get_ring(cee)->async_io(&io_uring_prep_send, timeout, ring_flags, fd, buf, len, io_flags);
}

ssize_t iouring_send_zc(int fd, const void* buf, size_t len, uint64_t flags, Timeout timeout, CascadingEventEngine* cee) {
    uint32_t io_flags = flags & 0xffffffff;
    uint32_t ring_flags = flags >> 32;
    return get_ring(cee)->async_io(&io_uring_prep_send_zc, timeout, ring_flags, fd, buf, len, io_flags, 0);
}

ssize_t iouring_sendmsg(int fd, const msghdr* msg, uint64_t flags, Timeout timeout, CascadingEventEngine* cee) {
    uint32_t io_flags = flags & 0xffffffff;
    uint32_t ring_flags = flags >> 32;
    return get_ring(cee)->async_io(&io_uring_prep_sendmsg, timeout, ring_flags, fd, msg, io_flags);
}

ssize_t iouring_sendmsg_zc(int fd, const msghdr* msg, uint64_t flags, Timeout timeout, CascadingEventEngine* cee) {
    uint32_t io_flags = flags & 0xffffffff;
    uint32_t ring_flags = flags >> 32;
    return get_ring(cee)->async_io(&io_uring_prep_sendmsg_zc, timeout, ring_flags, fd, msg, io_flags);
}

ssize_t iouring_recv(int fd, void* buf, size_t len, uint64_t flags, Timeout timeout, CascadingEventEngine* cee) {
    uint32_t io_flags = flags & 0xffffffff;
    uint32_t ring_flags = flags >> 32;
    return get_ring(cee)->async_io(&io_uring_prep_recv, timeout, ring_flags, fd, buf, len, io_flags);
}

ssize_t iouring_recvmsg(int fd, msghdr* msg, uint64_t flags, Timeout timeout, CascadingEventEngine* cee) {
    uint32_t io_flags = flags & 0xffffffff;
    uint32_t ring_flags = flags >> 32;
    return get_ring(cee)->async_io(&io_uring_prep_recvmsg, timeout, ring_flags, fd, msg, io_flags);
}

int iouring_connect(int fd, const sockaddr* addr, socklen_t addrlen, Timeout timeout, CascadingEventEngine* cee) {
    return get_ring(cee)->async_io(&io_uring_prep_connect, timeout, 0, fd, addr, addrlen);
}

int iouring_accept(int fd, sockaddr* addr, socklen_t* addrlen, Timeout timeout, CascadingEventEngine* cee) {
    return get_ring(cee)->async_io(&io_uring_prep_accept, timeout, 0, fd, addr, addrlen, 0);
}

int iouring_fsync(int fd, Timeout timeout, CascadingEventEngine* cee) {
    return get_ring(cee)->async_io(&io_uring_prep_fsync, timeout, 0, fd, 0);
}

int iouring_fdatasync(int fd, Timeout timeout, CascadingEventEngine* cee) {
    return get_ring(cee)->async_io(&io_uring_prep_fsync, timeout, 0, fd, IORING_FSYNC_DATASYNC);
}

int iouring_open(const char* path, int flags, mode_t mode, Timeout timeout, CascadingEventEngine* cee) {
    return get_ring(cee)->async_io(&io_uring_prep_openat, timeout, 0, AT_FDCWD, path, flags, mode);
}

int iouring_mkdir(const char* path, mode_t mode, Timeout timeout, CascadingEventEngine* cee) {
    return get_ring(cee)->async_io(&io_uring_prep_mkdirat, timeout, 0, AT_FDCWD, path, mode);
}

int iouring_close(int fd, Timeout timeout, CascadingEventEngine* cee) {
    return get_ring(cee)->async_io(&io_uring_prep_close, timeout, 0, fd);
}

bool iouring_register_files_enabled() {
    return iouringEngine::register_files_enabled();
}

int iouring_register_files(int fd, CascadingEventEngine* cee) {
    return get_ring(cee)->register_unregister_files(fd, true);
}

int iouring_unregister_files(int fd, CascadingEventEngine* cee) {
    return get_ring(cee)->register_unregister_files(fd, false);
}

void* new_iouring_event_engine(iouring_args args) {
    LOG_INFO("Init event engine: iouring ",
        make_named_value("is_master",     args.is_master),
        make_named_value("setup_sqpoll",  args.setup_sqpoll),
        make_named_value("setup_sq_aff",  args.setup_sq_aff),
        make_named_value("sq_thread_cpu", args.sq_thread_cpu));
    auto uring = NewObj<iouringEngine>() -> init(args);
    if (args.is_master) return uring;
    CascadingEventEngine* c = uring;
    return c;
}


}
